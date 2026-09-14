"""Progress measures and Fourier-space ablations from the paper (Section 5.1).

Two key progress measures are defined from the reverse-engineered algorithm:

* Restricted loss -- keep only the constant term and the terms corresponding to
  cos(w_k (a+b)) / sin(w_k (a+b)) for the key frequencies w_k; measure the
  cross-entropy of the resulting (ablated) logits.  Measures how well the
  *generalising circuit alone* solves the task -> drops during circuit
  formation, before test loss drops.

* Excluded loss  -- remove those key frequencies from the logits but keep
  everything else; measured on the *training* data.  The memorising solution is
  spread out in the Fourier domain so it is barely affected early on, but once
  the model relies on the Fourier circuit the excluded loss rises.

Plus the Gini coefficient of the Fourier-component norms of W_E and W_L (a
sparsity measure) and the sum of squared weights (paper Figure 7).
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# basic utilities
# --------------------------------------------------------------------------- #
def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a 1-D array (0 = maximally equal, 1 = maximally
    concentrated).  Used on the norms of the Fourier components of W_E / W_L."""
    v = np.asarray(values, dtype=float).ravel()
    v = np.abs(v)
    if v.sum() == 0:
        return 0.0
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) - (n + 1) * np.sum(v)) / (n * np.sum(v)))


def log_softmax_loss(logits_grid: np.ndarray, labels: np.ndarray,
                     flat_idx: Optional[np.ndarray] = None) -> float:
    """Mean cross-entropy of a [P, P, P] logit grid against labels.

    labels: [P, P] array of ground-truth outputs c(a, b).
    flat_idx: optional flat indices (over the P*P pairs) to average over, e.g.
    the training pairs.  If None, average over the whole grid.
    """
    P = logits_grid.shape[0]
    L = logits_grid.reshape(-1, P)                 # [P*P, P]
    y = labels.reshape(-1)                         # [P*P]
    # log-softmax in a numerically stable way
    m = L.max(axis=1, keepdims=True)
    logsum = np.log(np.exp(L - m).sum(axis=1)) + m[:, 0]
    ce = (logsum - L[np.arange(L.shape[0]), y])
    if flat_idx is None:
        return float(ce.mean())
    return float(ce[flat_idx].mean())


def _ablate_in_fourier(logits_grid: np.ndarray, keep2d: np.ndarray) -> np.ndarray:
    """Zero out Fourier components of the (a, b) axes not in the boolean keep
    mask and reconstruct logits.  Returns a real [P, P, P] array."""
    P = logits_grid.shape[0]
    F = np.fft.fft2(logits_grid, axes=(0, 1))
    F = F * keep2d[:, :, None].astype(np.complex128)
    return np.real(np.fft.ifft2(F, axes=(0, 1)))


# --------------------------------------------------------------------------- #
# key frequencies and Fourier masks
# --------------------------------------------------------------------------- #
def detect_key_freqs(norms_1d: np.ndarray, min_frac: float = 0.15,
                     exclude_dc: bool = True) -> list:
    """Detect the "key frequencies" of a matrix from its per-frequency norms.

    The norms array has one entry per frequency 0 .. P//2 (output of
    fourier_norms_1d).  Key frequencies are those (excluding DC unless
    requested) whose norm is a substantial fraction of the largest one; this
    mirrors the paper's "take the frequencies with nontrivial coefficients".
    """
    norms = np.asarray(norms_1d, dtype=float)
    freqs = np.arange(norms.size)
    if exclude_dc:
        norms = norms[1:]
        freqs = freqs[1:]
    if norms.size == 0 or norms.max() == 0:
        return []
    keep = np.nonzero(norms >= min_frac * norms.max())[0]
    return [int(freqs[i]) for i in keep]


def fourier_pair_mask(P: int, freqs: Sequence[int]) -> np.ndarray:
    """Boolean [P, P] mask of the 2D-Fourier components (k1, k2) belonging to
    one of the key frequencies, i.e. the four products
    {cos, sin}(w_k a) x {cos, sin}(w_k b) which appear as the DFT pairs
    (k, k), (k, P-k), (P-k, k), (P-k, P-k)."""
    mask = np.zeros((P, P), dtype=bool)
    for k in freqs:
        kk = int(k) % P
        if kk == 0 or kk * 2 == P:
            mask[kk, kk] = True
        else:
            mask[kk, kk] = True
            mask[kk, P - kk] = True
            mask[P - kk, kk] = True
            mask[P - kk, P - kk] = True
    return mask


def restricted_mask(P: int, freqs: Sequence[int]) -> np.ndarray:
    """Mask keeping only the DC term + the key-frequency products."""
    m = fourier_pair_mask(P, freqs)
    m[0, 0] = True
    return m


def excluded_mask(P: int, freqs: Sequence[int]) -> np.ndarray:
    """Mask removing the key-frequency products and keeping everything else."""
    return ~fourier_pair_mask(P, freqs)


# --------------------------------------------------------------------------- #
# the progress measures
# --------------------------------------------------------------------------- #
def restricted_loss(logits_grid: np.ndarray, labels: np.ndarray,
                    freqs: Sequence[int],
                    flat_idx: Optional[np.ndarray] = None) -> float:
    P = logits_grid.shape[0]
    m = restricted_mask(P, freqs)
    ablated = _ablate_in_fourier(logits_grid, m)
    return log_softmax_loss(ablated, labels, flat_idx)


def excluded_loss(logits_grid: np.ndarray, labels: np.ndarray,
                  freqs: Sequence[int],
                  flat_idx: Optional[np.ndarray] = None) -> float:
    P = logits_grid.shape[0]
    m = excluded_mask(P, freqs)
    ablated = _ablate_in_fourier(logits_grid, m)
    return log_softmax_loss(ablated, labels, flat_idx)


# --------------------------------------------------------------------------- #
# weight-norm style measures computed on a model snapshot
# --------------------------------------------------------------------------- #
def fourier_norms_matrix(mat: np.ndarray, axis: int = 0) -> np.ndarray:
    """Per-frequency norms of a real matrix along `axis` (see fourier.py)."""
    F = np.fft.rfft(mat, axis=axis)
    other = tuple(i for i in range(F.ndim) if i != axis)
    return np.sqrt(np.sum(np.abs(F) ** 2, axis=other))


def fourier_gini(mat: np.ndarray, axis: int = 0) -> float:
    """Gini coefficient of the Fourier-component norms of a matrix."""
    return gini_coefficient(fourier_norms_matrix(mat, axis=axis))
