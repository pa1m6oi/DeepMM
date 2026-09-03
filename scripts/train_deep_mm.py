from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from deep_mm.common.channel import sample_channels
from deep_mm.common.config import FixedMIMOConfig
from deep_mm.common.data import load_corpus, validate_training_corpus
from deep_mm.common.metrics import batched_secrecy_rate_torch
from deep_mm.models.deep_mm import DeepMM, build_training_targets
from deep_mm.models.checkpoints import build_metadata, save_checkpoint


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train fixed-channel Deep-MM.")
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--train-size", type=float, default=0.8)
    p.add_argument("--Nt", type=int, default=4)
    p.add_argument("--Nr", type=int, default=4)
    p.add_argument("--Ne", type=int, default=2)
    p.add_argument("--snr-db", type=float, default=15.0)
    p.add_argument("--L", "--layers", dest="L", type=int, default=16)
    p.add_argument("--learn-mode", choices=("direct", "offset", "scale", "net_direct"), default="net_direct")
    p.add_argument("--hidden-size", type=int, default=48)
    p.add_argument("--embed-dim", type=int, default=24)
    p.add_argument("--message-passing-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--grad-clip-norm", type=float, default=5.0)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--supervisor-epoch-num", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--reference-update-max-iter", type=int, default=100)
    p.add_argument("--reference-update-tol", type=float, default=1e-6)
    p.add_argument("--corpus", default=None)
    p.add_argument("--output", required=True)
    return p


def _loss_mse(Q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((Q.real - target.real) ** 2 + (Q.imag - target.imag) ** 2)


def _early_stopping_update(
    best_rate: float,
    best_epoch: int,
    stale_epochs: int,
    val_rate: float,
    epoch: int,
    min_delta: float,
):
    if val_rate > best_rate + min_delta:
        return val_rate, epoch, 0, True
    return best_rate, best_epoch, stale_epochs + 1, False


@torch.no_grad()
def _mean_rate(
    model: DeepMM,
    H: torch.Tensor,
    G: torch.Tensor,
    Pt: float,
    batch_size: int,
    inference_layers=None,
) -> float:
    model.eval()
    values = []
    for start in range(0, H.shape[0], batch_size):
        q, _ = model(
            H[start:start + batch_size],
            G[start:start + batch_size],
            Pt,
            inference_layers=inference_layers,
        )
        values.append(
            batched_secrecy_rate_torch(
                H[start:start + batch_size], G[start:start + batch_size], q
            )
        )
    model.train()
    return float(torch.cat(values).mean().item())


def _load_training_arrays(args, cfg):
    if getattr(args, "corpus", None):
        H_np, G_np, metadata = load_corpus(args.corpus)
        validate_training_corpus(H_np, G_np, metadata, cfg, expected_count=args.samples)
        return H_np, G_np, metadata
    rng = np.random.RandomState(args.seed + 1000)
    H_np, G_np = sample_channels(rng, cfg, args.samples)
    return H_np, G_np, None


def _metadata(args, cfg):
    return build_metadata(
        "deep_mm", args.Nt, args.Nr, args.Ne, args.snr_db, args.L,
        training_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=args.train_size,
        learning_rate=args.lr,
        reference_update_max_iter=args.reference_update_max_iter,
        supervisor_epoch_num=args.supervisor_epoch_num,
        default_inference_layers=6,
        learn_mode=str(getattr(args, "learn_mode", "net_direct")),
        grad_clip_norm=getattr(args, "grad_clip_norm", 5.0),
        seed=args.seed,
        early_stop_patience=getattr(args, "early_stop_patience", 3),
        early_stop_min_delta=getattr(args, "early_stop_min_delta", 1e-4),
    )


def train(args: argparse.Namespace) -> dict:
    if args.samples < 4 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("samples must be at least 4 and epochs/batch-size must be positive")
    if not 0.0 < args.train_size < 1.0:
        raise ValueError("train_size must satisfy 0 < train_size < 1")
    early_stop_patience = int(getattr(args, "early_stop_patience", 3))
    early_stop_min_delta = float(getattr(args, "early_stop_min_delta", 1e-4))
    grad_clip_norm = float(getattr(args, "grad_clip_norm", 5.0))
    if early_stop_patience < 0 or early_stop_min_delta < 0.0 or grad_clip_norm <= 0.0:
        raise ValueError("early-stop patience/min-delta must be non-negative and grad-clip-norm must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = FixedMIMOConfig(args.Nt, args.Nr, args.Ne, args.snr_db, args.seed)
    H_np, G_np, corpus_metadata = _load_training_arrays(args, cfg)
    train_count = int(args.train_size * len(H_np))
    val_count = (len(H_np) - train_count) // 2
    if train_count <= 0 or val_count <= 0:
        raise ValueError("train_size must leave non-empty training and validation sets")
    permutation = np.random.permutation(len(H_np))
    train_ids = permutation[:train_count]
    val_ids = permutation[train_count:train_count + val_count]
    device = torch.device(args.device)
    H = torch.as_tensor(H_np, dtype=torch.complex64, device=device)
    G = torch.as_tensor(G_np, dtype=torch.complex64, device=device)
    model = DeepMM(
        layers=args.L,
        hidden_size=args.hidden_size,
        embed_dim=args.embed_dim,
        learn_mode=str(getattr(args, "learn_mode", "net_direct")),
        scale_min=0.25,
        scale_max=4.0,
        nt_norm=16.0,
        message_passing_steps=args.message_passing_steps,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batches_per_epoch = int(math.ceil(train_count / float(args.batch_size)))
    supervisor_batches = int(args.supervisor_epoch_num * train_count / args.batch_size)
    global_step = 0
    best_rate = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    early_stopped = False
    stopped_epoch = None
    history = []
    train_indices = torch.as_tensor(train_ids, dtype=torch.long, device=device)
    val_indices = torch.as_tensor(val_ids, dtype=torch.long, device=device)
    for epoch in range(1, args.epochs + 1):
        order = train_indices[torch.randperm(train_count, device=device)]
        losses = []
        model.train()
        for start in range(0, train_count, args.batch_size):
            ids = order[start:start + args.batch_size]
            Hb, Gb = H[ids], G[ids]
            optimizer.zero_grad(set_to_none=True)
            Q, _ = model(Hb, Gb, cfg.Pt)
            if global_step < supervisor_batches:
                with torch.no_grad():
                    target, _ = build_training_targets(
                        Hb, Gb, cfg.Pt,
                        args.reference_update_max_iter,
                        args.reference_update_tol,
                    )
                loss = _loss_mse(Q, target)
            else:
                loss = -batched_secrecy_rate_torch(Hb, Gb, Q).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Deep-MM loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            global_step += 1
        val_rate = _mean_rate(model, H[val_indices], G[val_indices], cfg.Pt, args.batch_size)
        best_rate, best_epoch, stale_epochs, improved = _early_stopping_update(
            best_rate, best_epoch, stale_epochs, val_rate, epoch, early_stop_min_delta
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_rate": val_rate,
            "best_epoch": best_epoch,
            "early_stop_wait": stale_epochs,
        }
        history.append(record)
        if improved:
            save_checkpoint(args.output, model, _metadata(args, cfg), optimizer, epoch, record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            early_stopped = True
            stopped_epoch = epoch
            break
    metadata = _metadata(args, cfg)
    result = {
        "model": "Deep-MM", "output": str(Path(args.output)), "best_epoch": best_epoch,
        "best_val_rate": best_rate, "metadata": metadata, "corpus_metadata": corpus_metadata,
        "history": history, "early_stopped": early_stopped, "stopped_epoch": stopped_epoch,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    train(_parser().parse_args())


if __name__ == "__main__":
    main()

