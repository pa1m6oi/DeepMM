from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


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

