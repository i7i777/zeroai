"""Post-hoc reverse engineering of a trained model (paper Section 4).

Given a trained OneLayerTransformer this module computes the objects that the
paper analyses:

* W_E (embedding, [p, d]) and W_L = W_U @ W_out ([p, n_mlp]) Fourier norms and
  detected key frequencies.
* The attention pattern of each head from the final '=' token to token a,
  as a function of (a, b): [n_heads, P, P].
* The MLP neuron activations on the final token over the full input grid:
  [P, P, n_mlp].
* For each MLP neuron the best single-frequency degree-2 polynomial fit and its
  fraction of variance explained (paper Figure 5 left).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .fourier import fourier_norms_1d
from .measures import fourier_norms_matrix, detect_key_freqs
from .model import OneLayerTransformer


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def detect_key_freqs_of_model(model: OneLayerTransformer, min_frac: float = 0.15) -> list:
    """Key frequencies from the neuron-logit map W_L (paper: DFT on W_L over the
    logit axis, take frequencies with nontrivial coefficients)."""
    norms = fourier_norms_matrix(to_np(model.W_L), axis=0)   # per frequency 0..P//2
    return detect_key_freqs(norms, min_frac=min_frac)


@torch.no_grad()
def attention_pattern_grid(model: OneLayerTransformer,
                           head: int = 0) -> np.ndarray:
    """Attention paid by the final '=' token to token a for every (a, b):
    a [P, P] array (paper Figure 4 left / Figure 9 for all heads)."""
    a, b = model.all_pairs()
    P = model.p
    out = np.zeros((P, P))
    chunk = 1024
    for i in range(0, a.numel(), chunk):
        lg, diag = model.forward_full(a[i:i + chunk], b[i:i + chunk])
        pat = diag["attn_patterns"][head]              # [B, 3, 3]
        out[to_np(a[i:i + chunk]), to_np(b[i:i + chunk])] = to_np(pat[:, 2, 0])
    return out


@torch.no_grad()
def attention_patterns_all_heads(model: OneLayerTransformer) -> np.ndarray:
    """[n_heads, P, P] attention from '=' to a over the input grid."""
    a, b = model.all_pairs()
    P = model.p
    H = model.n_heads
    out = np.zeros((H, P, P))
    chunk = 1024
    for i in range(0, a.numel(), chunk):
        lg, diag = model.forward_full(a[i:i + chunk], b[i:i + chunk])
        aa = to_np(a[i:i + chunk])
        bb = to_np(b[i:i + chunk])
        for h in range(H):
            pat = diag["attn_patterns"][h]
            out[h, aa, bb] = to_np(pat[:, 2, 0])
    return out


@torch.no_grad()
def mlp_activations_grid(model: OneLayerTransformer) -> np.ndarray:
    """MLP ReLU activations on the final token over all (a, b): [P, P, n_mlp]."""
    a, b = model.all_pairs()
    P = model.p
    n = model.n_mlp
    out = np.zeros((P, P, n), dtype=np.float32)
    chunk = 512
    for i in range(0, a.numel(), chunk):
        lg, diag = model.forward_full(a[i:i + chunk], b[i:i + chunk])
        out[to_np(a[i:i + chunk]), to_np(b[i:i + chunk])] = to_np(diag["mlp_acts"])
    return out


# --------------------------------------------------------------------------- #
# per-neuron degree-2 single-frequency analysis (paper Section 4.3 / Fig. 5)
# --------------------------------------------------------------------------- #
def _cos_sin(p: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.arange(p, dtype=float)
    w = 2.0 * np.pi * k / p
    return np.cos(w * x), np.sin(w * x)


def single_freq_degree2_basis(p: int, k: int) -> np.ndarray:
    """Columns (one per basis function) of the degree-2 polynomial space in
    (cos/sin)(w_k a), (cos/sin)(w_k b), evaluated over the full (a, b) grid.
    Returns [P*P, 9]."""
    ca, sa = _cos_sin(p, k)
    cb, sb = _cos_sin(p, k)
    # outer products over the grid
    cols = [
        np.ones(p * p),
        np.repeat(ca, p), np.repeat(sa, p),
        np.tile(cb, p), np.tile(sb, p),
        (np.repeat(ca, p) * np.tile(cb, p)),
        (np.repeat(ca, p) * np.tile(sb, p)),
        (np.repeat(sa, p) * np.tile(cb, p)),
        (np.repeat(sa, p) * np.tile(sb, p)),
    ]
    return np.stack(cols, axis=1)


def neuron_frequency_fit(acts: np.ndarray, freqs: Sequence[int],
                         frac_threshold: float = 0.85,
                         chunk: int = 256) -> Dict:
    """Fit each MLP neuron's activation (over the [P, P] input grid) by the best
    degree-2 polynomial in a single candidate frequency.

    acts: [P, P, n_mlp] activations.
    Returns dict with per-neuron best frequency and fraction of variance
    explained (FVE), and a count of neurons with FVE above threshold per freq.
    """
    P = acts.shape[0]
    A = acts.reshape(-1, acts.shape[2]).astype(np.float64)   # [P*P, n]
    A = A - A.mean(axis=0, keepdims=True)
    denom = np.sum(A ** 2, axis=0)                           # [n]
    best_freq = np.zeros(acts.shape[2], dtype=int)
    best_fve = np.zeros(acts.shape[2])
    for k in freqs:
        X = single_freq_degree2_basis(P, k).astype(np.float64)
        X = X - X.mean(axis=0, keepdims=True)
        # project A onto the column space of X in chunks over neurons
        # normal equations: coef = (X^T X)^-1 X^T A
        XtX = X.T @ X
        XtA = X.T @ A
        try:
            coef = np.linalg.solve(XtX, XtA)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(XtX, XtA, rcond=None)[0]
        resid = A - X @ coef
        fve = 1.0 - np.sum(resid ** 2, axis=0) / np.maximum(denom, 1e-30)
        better = fve > best_fve
        best_fve[better] = fve[better]
        best_freq[better] = k
    # safety: reject poor fits
    best_freq[best_fve < 0] = -1

    counts = {}
    for k in freqs:
        counts[k] = int(np.sum((best_freq == k) & (best_fve >= frac_threshold)))
    counts["other/unfit"] = int(acts.shape[2] - sum(counts[k] for k in freqs))
    return {"best_freq": best_freq, "best_fve": best_fve, "counts": counts}


def wl_columns_for_cluster(W_L: np.ndarray, neuron_idx: np.ndarray) -> np.ndarray:
    """W_L columns (over the logit axis) for a cluster of neurons: [P, n_cluster]."""
    return W_L[:, neuron_idx]


def wl_fourier_heatmap(W_L_cols: np.ndarray) -> np.ndarray:
    """DFT along the logit axis of the W_L columns -> [P//2+1, n_cluster]
    magnitudes (paper Figure 5 right)."""
    return np.abs(np.fft.rfft(W_L_cols, axis=0))
