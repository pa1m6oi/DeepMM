from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .channel import sample_channels
from .config import FixedMIMOConfig


def _content_hash(H: np.ndarray, G: np.ndarray, metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(H).tobytes())
    digest.update(np.ascontiguousarray(G).tobytes())
    digest.update(json.dumps(dict(metadata), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _validate_distinct_channel_pairs(H: np.ndarray, G: np.ndarray) -> None:
    signatures = set()
    for h, g in zip(H, G):
        signature = (np.ascontiguousarray(h).tobytes(), np.ascontiguousarray(g).tobytes())
        if signature in signatures:
            raise ValueError("duplicate channel realization")
        signatures.add(signature)


def make_corpus(cfg: FixedMIMOConfig, count: int, seed: int, path: str) -> Dict[str, Any]:
    """Create or validate a reproducible direct-channel NPZ corpus."""
    if int(count) <= 0:
        raise ValueError("count must be positive")
    target = Path(path)
    requested = cfg.metadata()
    requested.update({"count": int(count), "seed": int(seed)})
    if target.exists():
        H, G, metadata = load_corpus(path)
        _validate_distinct_channel_pairs(H, G)
        for key, value in requested.items():
            if metadata.get(key) != value:
                raise ValueError("existing corpus metadata mismatch for {}".format(key))
        return metadata
    rng = np.random.RandomState(int(seed))
    H, G = sample_channels(rng, cfg, count)
    _validate_distinct_channel_pairs(H, G)
    metadata = dict(requested)
    metadata["content_hash"] = _content_hash(H, G, metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, H=H, G=G, metadata_json=json.dumps(metadata, sort_keys=True))
    tmp.replace(target)
    return metadata


def load_corpus(path: str):
    with np.load(path, allow_pickle=False) as data:
        H = np.asarray(data["H"])
        G = np.asarray(data["G"])
        metadata = json.loads(str(data["metadata_json"]))
    if H.ndim != 3 or G.ndim != 3 or H.shape[0] != G.shape[0] or H.shape[2] != G.shape[2]:
        raise ValueError("corpus arrays must have shapes (B,Nr,Nt) and (B,Ne,Nt)")
    expected = {key: value for key, value in metadata.items() if key != "content_hash"}
    if metadata.get("content_hash") != _content_hash(H, G, expected):
        raise ValueError("corpus content hash mismatch")
    return H, G, metadata


def validate_training_corpus(
    H: np.ndarray,
    G: np.ndarray,
    metadata: Mapping[str, Any],
    cfg: FixedMIMOConfig,
    expected_count: int = 50000,
) -> None:
    """Validate the fixed-channel training corpus contract."""
    expected = cfg.metadata()
    expected.update({"count": int(expected_count)})
    expected.pop("seed", None)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError("training corpus metadata mismatch for {}".format(key))
    if H.shape != (expected_count, cfg.Nr, cfg.Nt) or G.shape != (expected_count, cfg.Ne, cfg.Nt):
        raise ValueError("training corpus has an unexpected shape")
    _validate_distinct_channel_pairs(H, G)

