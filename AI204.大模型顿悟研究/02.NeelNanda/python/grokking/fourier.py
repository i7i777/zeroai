"""Fourier-analysis helpers (NumPy based).

The networks solve modular addition via the "Fourier multiplication"
algorithm: their weights and activations are sparse in the Fourier basis over
the P input values / P output logits.  These helpers compute the DFT norms used
throughout the paper (Figure 3, 4, 6, 7 ...).
"""
from __future__ import annotations

import numpy as np


def rfft_norms(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Magnitudes of the real-DFT components along `axis`.

    For an axis of length P this returns P//2 + 1 values (frequencies
    0 .. P//2).  Frequencies k and P-k are conjugate for real inputs, so they
    share a single magnitude (matching the paper's figures that plot only the
    first P//2 frequencies).
    """
    return np.abs(np.fft.rfft(x, axis=axis))


def fourier_norms_1d(mat: np.ndarray, axis: int = 0) -> np.ndarray:
    """L2 norm (over the non-transform axes) of each Fourier component.

    Used e.g. for the embedding matrix W_E (shape [P, d]): DFT along the value
    axis, then take the L2 norm over the residual-stream axis -> one scalar per
    frequency (paper Figure 3, left).
    """
    F = np.fft.rfft(mat, axis=axis)
    return np.sqrt(np.sum(np.abs(F) ** 2, axis=tuple(i for i in range(F.ndim) if i != axis)))


def fft2_logits(logits_grid: np.ndarray, axes=(0, 1)) -> np.ndarray:
    """2D DFT of a [P, P, P] logit grid over the (a, b) input axes."""
    return np.fft.fft2(logits_grid, axes=axes)


def logits_fourier_norm(logits_grid: np.ndarray) -> np.ndarray:
    """Norm over the output (logit) axis of the 2D-DFT coefficients.

    Paper Figure 4 (right): the logits of a generalising model, represented in
    the 2D Fourier basis over the inputs, have 20 significant components (the
    sines/cosines of the five key frequencies).
    """
    F = fft2_logits(logits_grid)                       # [P, P, P] complex
    return np.sqrt(np.sum(np.abs(F) ** 2, axis=2))     # [P, P]


def logits_fourier_norms_1d(logits_grid: np.ndarray) -> np.ndarray:
    """Fourier norms of the logits collapsed to one input axis.

    Returns magnitudes per frequency over the input axis (used to sanity-check
    sparsity of logits in the input Fourier basis).
    """
    F = np.fft.rfft(logits_grid, axis=0)               # over a
    return np.sqrt(np.sum(np.abs(F) ** 2, axis=(1, 2)))
