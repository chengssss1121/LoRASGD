"""Exact numerical factor metrics for a single LoRA pair.

All reported norms are matrix norms. In particular, ``spectral_sum`` is
``||A||_2**2 + ||B||_2**2``, not a Frobenius norm and not the spectral norm
of a stacked factor matrix. Inputs are never modified.
"""

from __future__ import annotations

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def spectral_norm_sq(matrix: np.ndarray) -> float:
    """Return the largest singular value squared, using a full SVD calculation.

    This is an exact numerical calculation (up to floating-point error), rather
    than a power-iteration estimate. Computation is performed in float64.
    """
    array = _matrix(matrix, "matrix")
    largest = float(np.linalg.svd(array, compute_uv=False)[0])
    return largest * largest


def pair_metrics(A: np.ndarray, B: np.ndarray) -> dict[str, float | int]:
    """Measure one pair with shapes A=(r, in), B=(out, r).

    Product norms use the small core ``R_B @ R_A.T`` obtained from reduced QR
    factorizations of B and A.T. Thus an out-by-in product need not be formed.
    ``rank`` is the configured adapter width r, not an estimated numerical rank.
    """
    A = _matrix(A, "A")
    B = _matrix(B, "B")
    if A.shape[0] != B.shape[1]:
        raise ValueError(f"Incompatible LoRA factors: A{A.shape}, B{B.shape}")

    a_spectral_sq = spectral_norm_sq(A)
    b_spectral_sq = spectral_norm_sq(B)
    a_frobenius_sq = float(np.sum(A * A))
    b_frobenius_sq = float(np.sum(B * B))
    _, R_B = np.linalg.qr(B, mode="reduced")
    _, R_A = np.linalg.qr(A.T, mode="reduced")
    core = R_B @ R_A.T
    product_singular_values = np.linalg.svd(core, compute_uv=False)
    imbalance = A @ A.T - B.T @ B
    imbalance = 0.5 * (imbalance + imbalance.T)

    return {
        "a_spectral_sq": a_spectral_sq,
        "b_spectral_sq": b_spectral_sq,
        "spectral_sum": a_spectral_sq + b_spectral_sq,
        "a_frobenius_sq": a_frobenius_sq,
        "b_frobenius_sq": b_frobenius_sq,
        "frobenius_sum": a_frobenius_sq + b_frobenius_sq,
        "product_spectral": float(product_singular_values[0]),
        "product_frobenius": float(np.linalg.norm(core, ord="fro")),
        "imbalance_spectral": float(np.max(np.abs(np.linalg.eigvalsh(imbalance)))),
        "rank": int(A.shape[0]),
    }
