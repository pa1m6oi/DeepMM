from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import torch


def hermitian_np(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.conj().T)


def log2det_np(matrix: np.ndarray) -> float:
    matrix = hermitian_np(matrix)
    sign, value = np.linalg.slogdet(matrix)
    if np.real(sign) <= 0:
        jitter = 1e-10 * np.eye(matrix.shape[0], dtype=np.complex128)
        sign, value = np.linalg.slogdet(matrix + jitter)
    if np.real(sign) <= 0 or not np.isfinite(value):
        raise FloatingPointError("log-det evaluation returned a non-finite value")
    return float(np.real(value) / math.log(2.0))


def secrecy_rate_np(H: np.ndarray, G: np.ndarray, Q: np.ndarray) -> float:
    H = np.asarray(H, dtype=np.complex128)
    G = np.asarray(G, dtype=np.complex128)
    Q = hermitian_np(Q)
    if H.ndim != 2 or G.ndim != 2 or Q.ndim != 2:
        raise ValueError("H, G, and Q must be matrices")
    if H.shape[1] != G.shape[1] or H.shape[1] != Q.shape[0] or Q.shape[0] != Q.shape[1]:
        raise ValueError("H, G, and Q dimensions are inconsistent")
    bob = np.eye(H.shape[0], dtype=np.complex128) + H @ Q @ H.conj().T
    eve = np.eye(G.shape[0], dtype=np.complex128) + G @ Q @ G.conj().T
    return log2det_np(bob) - log2det_np(eve)


def covariance_diagnostics(Q: np.ndarray, Pt: float, tol: float = 1e-5) -> Dict[str, Any]:
    Q = hermitian_np(Q)
    eigenvalues = np.linalg.eigvalsh(Q).real
    trace = float(np.real(np.trace(Q)))
    return {
        "hermitian_error": float(np.linalg.norm(Q - Q.conj().T)),
        "min_eigenvalue": float(np.min(eigenvalues)),
        "trace": trace,
        "psd": bool(np.min(eigenvalues) >= -tol),
        "trace_feasible": bool(trace <= float(Pt) + tol),
        "feasible": bool(np.min(eigenvalues) >= -tol and trace <= float(Pt) + tol),
    }


def hermitian_torch(matrix: torch.Tensor) -> torch.Tensor:
    """Return the Hermitian part of a complex matrix or matrix batch."""
    return 0.5 * (matrix + matrix.conj().transpose(-2, -1))


def _torch_eye(batch_size: int, size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.eye(size, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, size, size)


def log2det_torch(matrix: torch.Tensor) -> torch.Tensor:
    """Evaluate the base-2 log-determinant of positive-definite matrices."""
    sign, value = torch.linalg.slogdet(matrix)
    if torch.any(sign.real <= 0):
        eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        sign, value = torch.linalg.slogdet(matrix + 1e-12 * eye)
    return value.real / math.log(2.0)


def secrecy_rate_torch(H: torch.Tensor, G: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Compute the secrecy rate for one direct-channel realization."""
    if H.ndim != 2 or G.ndim != 2 or Q.ndim != 2:
        raise ValueError("single-channel secrecy_rate expects matrices")
    if H.shape[1] != G.shape[1] or Q.shape != (H.shape[1], H.shape[1]):
        raise ValueError("inconsistent H, G, and Q dimensions")
    id_matrix = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    ie_matrix = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return log2det_torch(id_matrix + H @ Q @ H.conj().T) - log2det_torch(ie_matrix + G @ Q @ G.conj().T)


def batched_secrecy_rate_torch(H: torch.Tensor, G: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Compute secrecy rates for a same-shaped batch of channels."""
    if H.ndim != 3 or G.ndim != 3 or Q.ndim != 3:
        raise ValueError("batched secrecy_rate expects rank-three tensors")
    if H.shape[0] != G.shape[0] or H.shape[0] != Q.shape[0] or H.shape[2] != G.shape[2]:
        raise ValueError("inconsistent batched channel dimensions")
    batch_size, nr, _ = H.shape
    ne = G.shape[1]
    id_matrix = _torch_eye(batch_size, nr, H.dtype, H.device)
    ie_matrix = _torch_eye(batch_size, ne, G.dtype, G.device)
    bob = log2det_torch(id_matrix + H @ Q @ H.conj().transpose(-2, -1))
    eve = log2det_torch(ie_matrix + G @ Q @ G.conj().transpose(-2, -1))
    return bob - eve

