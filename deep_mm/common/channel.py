from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import FixedMIMOConfig


def _sample_complex_gaussian(
    rng: np.random.RandomState,
    shape: Tuple[int, ...],
    power: float,
) -> np.ndarray:
    scale = np.sqrt(float(power) / 2.0)
    return scale * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )


def validate_channel_pair(H: np.ndarray, G: np.ndarray) -> None:
    """Validate direct fixed-channel shapes for one realization."""
    H = np.asarray(H)
    G = np.asarray(G)
    if H.ndim != 2 or G.ndim != 2:
        raise ValueError("H and G must be two-dimensional arrays")
    if H.shape[1] != G.shape[1]:
        raise ValueError("H and G must share the transmit dimension")
    if H.shape[0] <= 0 or G.shape[0] <= 0 or H.shape[1] <= 0:
        raise ValueError("channel dimensions must be positive")
    if not np.iscomplexobj(H) or not np.iscomplexobj(G):
        raise ValueError("H and G must be complex-valued")


def sample_channel(
    rng: np.random.RandomState,
    cfg: FixedMIMOConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample one independent fixed-antenna ideal-CSI channel pair."""
    H = _sample_complex_gaussian(
        rng, (int(cfg.Nr), int(cfg.Nt)), cfg.channel_power
    ).astype(np.complex128)
    G = _sample_complex_gaussian(
        rng, (int(cfg.Ne), int(cfg.Nt)), cfg.channel_power
    ).astype(np.complex128)
    return H, G


def sample_channels(
    rng: np.random.RandomState,
    cfg: FixedMIMOConfig,
    count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a batch of direct fixed-antenna channel pairs."""
    if int(count) <= 0:
        raise ValueError("count must be positive")
    H = _sample_complex_gaussian(
        rng, (int(count), int(cfg.Nr), int(cfg.Nt)), cfg.channel_power
    ).astype(np.complex128)
    G = _sample_complex_gaussian(
        rng, (int(count), int(cfg.Ne), int(cfg.Nt)), cfg.channel_power
    ).astype(np.complex128)
    return H, G

