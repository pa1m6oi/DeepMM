from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch

from deep_mm.common.metrics import secrecy_rate_np


def make_hermitian(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.conj().transpose(-2, -1))


def _eye(batch_size: int, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(size, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, size, size)


def log2det(matrix: torch.Tensor) -> torch.Tensor:
    sign, value = torch.linalg.slogdet(matrix)
    if torch.any(sign.real <= 0):
        eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        sign, value = torch.linalg.slogdet(matrix + 1e-12 * eye)
    return value.real / math.log(2.0)


def secrecy_rate(H: torch.Tensor, G: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    if H.ndim != 2 or G.ndim != 2 or Q.ndim != 2:
        raise ValueError("single-channel secrecy_rate expects matrices")
    if H.shape[1] != G.shape[1] or Q.shape != (H.shape[1], H.shape[1]):
        raise ValueError("inconsistent H, G, and Q dimensions")
    Id = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    Ie = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return log2det(Id + H @ Q @ H.conj().T) - log2det(Ie + G @ Q @ G.conj().T)


def batched_secrecy_rate(H: torch.Tensor, G: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    if H.ndim != 3 or G.ndim != 3 or Q.ndim != 3:
        raise ValueError("batched secrecy_rate expects rank-three tensors")
    if H.shape[0] != G.shape[0] or H.shape[0] != Q.shape[0] or H.shape[2] != G.shape[2]:
        raise ValueError("inconsistent batched channel dimensions")
    batch_size, nr, _ = H.shape
    ne = G.shape[1]
    Id = _eye(batch_size, nr, H.dtype, H.device)
    Ie = _eye(batch_size, ne, G.dtype, G.device)
    bob = log2det(Id + H @ Q @ H.conj().transpose(-2, -1))
    eve = log2det(Ie + G @ Q @ G.conj().transpose(-2, -1))
    return bob - eve


@torch.no_grad()
def gamma_from_mmcf(H: torch.Tensor, Pt: float) -> torch.Tensor:
    if H.ndim not in (2, 3):
        raise ValueError("H must have rank 2 or 3")
    if float(Pt) <= 0:
        raise ValueError("Pt must be positive")
    single = H.ndim == 2
    H_batch = H.unsqueeze(0) if single else H
    gram = make_hermitian(H_batch.conj().transpose(-2, -1) @ H_batch)
    spectral_value = torch.linalg.eigvalsh(gram).real.max(dim=-1).values
    delta = min(0.01 / float(Pt), 0.5)
    gamma = (float(delta) * spectral_value.square()).clamp_min(1e-12)
    return gamma[0] if single else gamma


def _project_trace(Q_tilde: torch.Tensor, Pt: float) -> torch.Tensor:
    trace = torch.diagonal(Q_tilde, dim1=-2, dim2=-1).sum(dim=-1).real
    scale = torch.where(
        trace <= 0,
        torch.zeros_like(trace),
        torch.where(
            trace <= Pt,
            torch.ones_like(trace),
            torch.as_tensor(Pt, dtype=trace.dtype, device=trace.device) / (trace + 1e-12),
        ),
    )
    return Q_tilde * scale.reshape((-1,) + (1,) * (Q_tilde.ndim - 1)).to(Q_tilde.dtype)


def mmcf_terms(Q: torch.Tensor, H: torch.Tensor, G: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the legitimate and eavesdropper resolvent-gradient terms."""
    if H.ndim == 2:
        H = H.unsqueeze(0)
        G = G.unsqueeze(0)
        Q = Q.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    batch_size, nr, _ = H.shape
    ne = G.shape[1]
    Id = _eye(batch_size, nr, H.dtype, H.device)
    Ie = _eye(batch_size, ne, G.dtype, G.device)
    Xd = Id + H @ Q @ H.conj().transpose(-2, -1)
    Xe = Ie + G @ Q @ G.conj().transpose(-2, -1)
    H_bar = H.conj().transpose(-2, -1) @ torch.linalg.inv(Xd) @ H
    G_bar = G.conj().transpose(-2, -1) @ torch.linalg.inv(Xe) @ G
    if squeeze:
        return H_bar[0], G_bar[0]
    return H_bar, G_bar


def mmcf_direction(
    Q: torch.Tensor,
    H_bar: torch.Tensor,
    G_bar: torch.Tensor,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma_value = gamma.real.to(dtype=Q.real.dtype, device=Q.device)
    if Q.ndim == 2:
        F = make_hermitian(H_bar - G_bar + gamma_value.to(Q.dtype) * (Q + Q.conj().T))
    else:
        F = make_hermitian(
            H_bar - G_bar
            + gamma_value.reshape(-1, 1, 1).to(Q.dtype)
            * (Q + Q.conj().transpose(-2, -1))
        )
    eigenvalues, vectors = torch.linalg.eigh(F)
    positive = torch.relu(eigenvalues.real)
    if Q.ndim == 2:
        direction = (vectors * positive) @ vectors.conj().T
    else:
        direction = (vectors * positive.unsqueeze(-2)) @ vectors.conj().transpose(-2, -1)
    return F, make_hermitian(direction)


def mmcf_step(Q: torch.Tensor, H: torch.Tensor, G: torch.Tensor, Pt: float, gamma: torch.Tensor) -> torch.Tensor:
    H_bar, G_bar = mmcf_terms(Q, H, G)
    _F, direction = mmcf_direction(Q, H_bar, G_bar, gamma)
    denominator = 2.0 * gamma.real.to(dtype=direction.real.dtype, device=direction.device) + 1e-12
    if direction.ndim == 2:
        candidate = Q + direction / denominator.to(direction.dtype)
    else:
        candidate = Q + direction / denominator.reshape(-1, 1, 1).to(direction.dtype)
    if candidate.ndim == 2:
        return make_hermitian(_project_trace(candidate.unsqueeze(0), Pt)[0])
    return make_hermitian(_project_trace(candidate, Pt))


@torch.no_grad()
def run_mmcf(
    H: torch.Tensor,
    G: torch.Tensor,
    Pt: float,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_trace: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Run projected MM-CF on one direct fixed-channel pair."""
    if H.ndim != 2 or G.ndim != 2:
        raise ValueError("run_mmcf expects H=(Nr,Nt), G=(Ne,Nt)")
    nt = H.shape[1]
    Q = (float(Pt) / nt) * torch.eye(nt, dtype=H.dtype, device=H.device)
    gamma = gamma_from_mmcf(H.unsqueeze(0), Pt)[0]
    initial_rate = float(secrecy_rate(H, G, Q).item())
    rates = [initial_rate]
    converged = False
    for _ in range(int(max_iter)):
        candidate = mmcf_step(Q, H, G, Pt, gamma)
        rate = float(secrecy_rate(H, G, candidate).item())
        Q = candidate
        rates.append(rate)
        if tol >= 0 and abs(rates[-1] - rates[-2]) <= float(tol):
            converged = True
            break
    trace = {
        "rates": rates if return_trace else [rates[-1]],
        "iterations": len(rates) - 1,
        "converged": converged,
        "trace_valid_length": len(rates),
        "gamma": float(gamma.item()),
    }
    return Q, trace


@torch.no_grad()
def run_mmcf_batch(
    H: torch.Tensor,
    G: torch.Tensor,
    Pt: float,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_trace: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Run projected MM-CF independently for a same-shaped channel batch."""
    if H.ndim != 3 or G.ndim != 3:
        raise ValueError("run_mmcf_batch expects H=(B,Nr,Nt), G=(B,Ne,Nt)")
    if H.shape[0] != G.shape[0] or H.shape[2] != G.shape[2]:
        raise ValueError("H and G batch size and Nt must match")
    gamma = gamma_from_mmcf(H, Pt)
    batch_size, _nr, nt = H.shape
    Q = (float(Pt) / nt) * _eye(batch_size, nt, H.dtype, H.device).clone()
    rates = [batched_secrecy_rate(H, G, Q)]
    active = torch.ones(batch_size, dtype=torch.bool, device=H.device)
    converged = torch.zeros(batch_size, dtype=torch.bool, device=H.device)
    for _ in range(int(max_iter)):
        candidate = mmcf_step(Q, H, G, Pt, gamma)
        proposed = batched_secrecy_rate(H, G, candidate)
        Q = torch.where(active[:, None, None], candidate, Q)
        previous = rates[-1]
        current = torch.where(active, proposed, previous)
        rates.append(current)
        if tol >= 0:
            newly = active & (torch.abs(current - previous) <= float(tol))
            converged = converged | newly
            active = active & ~newly
            if not torch.any(active):
                break
    trace = {
        "rates": torch.stack(rates, dim=1) if return_trace else rates[-1],
        "iterations": len(rates) - 1,
        "converged": converged,
        "trace_valid_length": len(rates),
        "gamma": gamma,
    }
    return Q, trace


def numpy_mmcf(H: np.ndarray, G: np.ndarray, Pt: float, max_iter: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, Dict]:
    """Convenience wrapper for timing/aggregation code."""
    H_t = torch.as_tensor(H, dtype=torch.complex64)
    G_t = torch.as_tensor(G, dtype=torch.complex64)
    Q_t, trace = run_mmcf(H_t, G_t, Pt, max_iter=max_iter, tol=tol, return_trace=True)
    Q = Q_t.cpu().numpy().astype(np.complex128, copy=False)
    trace["rate"] = float(secrecy_rate_np(H, G, Q))
    return Q, trace

