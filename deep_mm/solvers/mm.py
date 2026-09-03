"""Safeguarded water-filling MM solver.

This module contains the original iterative MM reference method for the
fixed-channel project. It keeps the water-filling update but uses a rate-based
majorizer backtracking safeguard so numerical two-cycles are not reported as
the final solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

from deep_mm.common.metrics import hermitian_torch, secrecy_rate_np, secrecy_rate_torch


TensorLike = Union[np.ndarray, torch.Tensor]


def _to_complex_tensor(value: TensorLike, device: Union[torch.device, str] = "cpu") -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.complex64)
    return torch.as_tensor(value, dtype=torch.complex64, device=device)


def _validate_inputs(H: torch.Tensor, G: torch.Tensor, Pt: float, max_iter: int) -> None:
    if H.ndim != 2 or G.ndim != 2:
        raise ValueError("H and G must be rank-two channel matrices")
    if H.shape[0] <= 0 or G.shape[0] <= 0 or H.shape[1] <= 0:
        raise ValueError("H and G must have positive dimensions")
    if H.shape[1] != G.shape[1]:
        raise ValueError("H and G must have compatible transmit dimensions")
    if not np.isfinite(float(Pt)) or float(Pt) <= 0:
        raise ValueError("Pt must be positive and finite")
    if int(max_iter) <= 0:
        raise ValueError("max_iter must be positive")


def _mm_candidate(
    Q: torch.Tensor,
    H: torch.Tensor,
    G: torch.Tensor,
    Pt: float,
    gamma: float,
    id_matrix: torch.Tensor,
    ie_matrix: torch.Tensor,
    it_matrix: torch.Tensor,
) -> torch.Tensor:
    """Construct one water-filling MM candidate for a fixed majorizer."""
    gamma_tensor = torch.as_tensor(gamma, dtype=H.real.dtype, device=H.device)
    M = -gamma_tensor.to(dtype=H.dtype) * it_matrix
    xd = id_matrix + H @ Q @ H.mH
    xe = ie_matrix + G @ Q @ G.mH
    A = H.mH @ torch.linalg.solve(xd, H)
    L = G.mH @ torch.linalg.solve(xe, G)
    J = A - Q.mH @ M - M @ Q
    F = hermitian_torch(J - L)

    eigvals, eigvecs = torch.linalg.eigh(F)
    zeta = waterfill_mu(eigvals, 2.0 * gamma_tensor * float(Pt))
    new_eigs = torch.relu(eigvals.real - zeta)
    candidate = (eigvecs * (new_eigs / (2.0 * gamma_tensor)).to(dtype=eigvecs.dtype)) @ eigvecs.mH
    return hermitian_torch(candidate)


def waterfill_mu(eigs: torch.Tensor, target_sum: Union[float, torch.Tensor]) -> torch.Tensor:
    """Solve ``sum(max(eigs - mu, 0)) = target_sum`` by active-set sorting."""

    eigs = eigs.real
    if eigs.ndim != 1 or eigs.numel() == 0:
        raise ValueError("eigs must be a non-empty vector")
    target = torch.as_tensor(target_sum, dtype=eigs.dtype, device=eigs.device)
    if float(target.item()) <= 0.0:
        return eigs.max()

    sorted_eigs = torch.sort(eigs, descending=True).values
    counts = torch.arange(1, sorted_eigs.numel() + 1, dtype=eigs.dtype, device=eigs.device)
    mu_candidates = (torch.cumsum(sorted_eigs, dim=0) - target) / counts
    active = sorted_eigs > mu_candidates
    last_active = int(torch.nonzero(active, as_tuple=False)[-1].item()) if torch.any(active) else 0
    return mu_candidates[last_active]


@torch.no_grad()
def _run_original_mm(
    H: TensorLike,
    G: TensorLike,
    Pt: float,
    sigma_d: float = 1.0,
    sigma_e: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100,
    device: Union[torch.device, str] = "cpu",
    record_history: bool = False,
):
    if not np.isfinite(float(tol)) or float(tol) < 0:
        raise ValueError("tol must be finite and non-negative")
    if not np.isfinite(float(sigma_d)) or float(sigma_d) <= 0:
        raise ValueError("sigma_d must be positive and finite")
    if not np.isfinite(float(sigma_e)) or float(sigma_e) <= 0:
        raise ValueError("sigma_e must be positive and finite")

    H_t = _to_complex_tensor(H, device=device) / float(sigma_d)
    G_t = _to_complex_tensor(G, device=device) / float(sigma_e)
    _validate_inputs(H_t, G_t, Pt, max_iter)
    nd, nt = H_t.shape
    ne = G_t.shape[0]

    id_matrix = torch.eye(nd, dtype=H_t.dtype, device=H_t.device)
    ie_matrix = torch.eye(ne, dtype=H_t.dtype, device=H_t.device)
    it_matrix = torch.eye(nt, dtype=H_t.dtype, device=H_t.device)
    Q = (float(Pt) / nt) * it_matrix

    delta = min(0.01 / float(Pt), 0.5)
    lam_max = torch.linalg.eigvalsh(hermitian_torch(H_t.mH @ H_t)).real.max()
    gamma = torch.clamp(delta * lam_max.square(), min=1e-12)
    current_rate = float(secrecy_rate_torch(H_t, G_t, Q).item())
    best_Q = Q.clone()
    best_rate = current_rate
    rate_history = []
    converged = False
    status = "max_iter"
    backtracking_history = []

    for _ in range(int(max_iter)):
        trial_gamma = float(gamma.item())
        accepted = False
        backtracks = 0
        for _ in range(50):
            candidate = _mm_candidate(
                Q, H_t, G_t, Pt, trial_gamma, id_matrix, ie_matrix, it_matrix
            )
            candidate_rate = float(secrecy_rate_torch(H_t, G_t, candidate).item())
            if candidate_rate + 1e-7 >= current_rate:
                accepted = True
                break
            trial_gamma *= 2.0
            backtracks += 1

        if not accepted:
            status = "backtracking_failed"
            break

        Q = candidate
        gamma = torch.as_tensor(trial_gamma, dtype=gamma.dtype, device=gamma.device)
        previous_rate = current_rate
        current_rate = candidate_rate
        if current_rate >= best_rate:
            best_Q = Q.clone()
            best_rate = current_rate
        rate_history.append(current_rate)
        backtracking_history.append(backtracks)

        if abs(current_rate - previous_rate) <= float(tol):
            converged = True
            status = "converged"
            break

    Q = best_Q

    trace_q = torch.real(torch.trace(Q))
    if float(trace_q.item()) > 0.0:
        Q = hermitian_torch(Q * (float(Pt) / (trace_q + 1e-12)))
    rate = float(secrecy_rate_torch(H_t, G_t, Q).item())
    if record_history and rate_history:
        rate_history[-1] = rate
    return Q, rate, float(gamma.item()), rate_history, converged, status, backtracking_history


@torch.no_grad()
def solve_original_mm(
    H: TensorLike,
    G: TensorLike,
    Pt: float,
    sigma_d: float = 1.0,
    sigma_e: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100,
    device: Union[torch.device, str] = "cpu",
):
    """Return the covariance produced by the original MM solver."""

    Q, rate, _gamma, _history, _converged, _status, _backtracking = _run_original_mm(
        H, G, Pt, sigma_d=sigma_d, sigma_e=sigma_e, tol=tol,
        max_iter=max_iter, device=device,
    )
    return Q, rate


@torch.no_grad()
def solve_original_mm_with_history(
    H: TensorLike,
    G: TensorLike,
    Pt: float,
    sigma_d: float = 1.0,
    sigma_e: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100,
    device: Union[torch.device, str] = "cpu",
):
    """Return the original MM covariance and per-iteration history."""

    result = _run_original_mm(
        H, G, Pt, sigma_d=sigma_d, sigma_e=sigma_e, tol=tol,
        max_iter=max_iter, device=device, record_history=True,
    )
    return result[:4]


@torch.no_grad()
def run_mm(
    H: TensorLike,
    G: TensorLike,
    Pt: float,
    max_iter: int = 100,
    tol: float = 1e-6,
    return_trace: bool = False,
    device: Union[torch.device, str] = "cpu",
):
    """Run the safeguarded water-filling MM solver with a project trace."""

    Q, rate, gamma, history, converged, status, backtracking = _run_original_mm(
        H, G, Pt, tol=tol, max_iter=max_iter, device=device,
        record_history=return_trace,
    )
    rates = history if return_trace else [rate]
    return Q, {
        "rates": rates,
        "iterations": len(history),
        "converged": converged,
        "status": status,
        "trace_valid_length": len(rates),
        "gamma": gamma,
        "backtracking_steps": backtracking if return_trace else (backtracking[-1] if backtracking else 0),
        "rate": rate,
    }


def numpy_mm(
    H: np.ndarray,
    G: np.ndarray,
    Pt: float,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    """Return the original MM covariance in the evaluator's NumPy format."""

    Q_t, trace = run_mm(H, G, Pt, max_iter=max_iter, tol=tol, return_trace=True)
    Q = Q_t.detach().cpu().numpy().astype(np.complex128, copy=False)
    trace["rate"] = float(secrecy_rate_np(H, G, Q))
    return Q, trace


@dataclass
class OriginalMMBeamformingEvaluator:
    Pt: float
    max_iter: int = 100
    tol: float = 1e-6
    device: str = "cpu"

    def __call__(self, H: np.ndarray, G: np.ndarray):
        return solve_original_mm(
            H, G, self.Pt, max_iter=self.max_iter, tol=self.tol, device=self.device
        )

