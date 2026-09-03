from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from deep_mm.common.channel import sample_channel
from deep_mm.common.config import FixedMIMOConfig
from deep_mm.common.metrics import covariance_diagnostics, secrecy_rate_np
from deep_mm.models.checkpoints import load_checkpoint
from deep_mm.models.deep_mm import DeepMM


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a released Deep-MM checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/deep_mm_nt4_nr4_ne2_L06.pt"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = torch.load(args.checkpoint, map_location=args.device)
    recorded_config = payload["model_config"]
    model_config = {
        key: recorded_config[key]
        for key in (
            "layers",
            "hidden_size",
            "embed_dim",
            "learn_mode",
            "scale_min",
            "scale_max",
            "nt_norm",
            "message_passing_steps",
        )
    }
    metadata = payload["metadata"]
    model = DeepMM(**model_config).to(args.device)
    load_checkpoint(str(args.checkpoint), model, device=args.device, expected=metadata)
    model.eval()

    cfg = FixedMIMOConfig(
        Nt=int(metadata["Nt"]),
        Nr=int(metadata["Nr"]),
        Ne=int(metadata["Ne"]),
        snr_db=float(metadata["snr_db"]),
        seed=args.seed,
    )
    H, G = sample_channel(np.random.RandomState(args.seed), cfg)
    Ht = torch.as_tensor(H, dtype=torch.complex64, device=args.device).unsqueeze(0)
    Gt = torch.as_tensor(G, dtype=torch.complex64, device=args.device).unsqueeze(0)
    with torch.no_grad():
        Q, gamma = model(Ht, Gt, cfg.Pt, inference_layers=int(metadata["default_inference_layers"]))
    Q_np = Q[0].cpu().numpy()
    result = {
        "checkpoint": str(args.checkpoint),
        "secrecy_rate_bit_per_s": float(secrecy_rate_np(H, G, Q_np)),
        "covariance_diagnostics": covariance_diagnostics(Q_np, cfg.Pt),
        "gamma": gamma[0].cpu().numpy().tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

