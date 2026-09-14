"""Matplotlib rendering of the paper's main figures from run outputs.

Each function takes primitives (arrays / metrics DataFrames) so it can be used
both for single runs and averaged over seeds.  See make_figures.py for the CLI
that drives these functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fourier import fourier_norms_1d, logits_fourier_norm, logits_fourier_norms_1d
from .measures import (excluded_loss, restricted_loss, excluded_mask,
                       restricted_mask, fourier_pair_mask)

# --------------------------------------------------------------------------- #
# Figure 2: grokking curves (train/test accuracy & loss vs epoch)
# --------------------------------------------------------------------------- #
def fig2_grokking_curves(metrics_list: Sequence[pd.DataFrame],
                         path: Path, losses: Sequence[str] = ("linear", "log")) -> List[Path]:
    """metrics_list: one metrics.csv per seed (epoch, train_loss, train_acc,
    test_loss, test_acc).  Plots per-seed translucent + mean bold."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    out = []
    for scale in losses:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for m in metrics_list:
            ep = m["epoch"].values
            axes[0].plot(ep, m["train_acc"], color="tab:blue", alpha=0.25)
            axes[0].plot(ep, m["test_acc"], color="tab:red", alpha=0.25)
            if scale == "log":
                axes[1].semilogy(ep, m["train_loss"], color="tab:blue", alpha=0.25)
                axes[1].semilogy(ep, m["test_loss"], color="tab:red", alpha=0.25)
            else:
                axes[1].plot(ep, m["train_loss"], color="tab:blue", alpha=0.25)
                axes[1].plot(ep, m["test_loss"], color="tab:red", alpha=0.25)
        # mean
        alldf = pd.concat(metrics_list).groupby("epoch").mean()
        ep = alldf.index.values
        axes[0].plot(ep, alldf["train_acc"], color="tab:blue", lw=2.5, label="Train")
        axes[0].plot(ep, alldf["test_acc"], color="tab:red", lw=2.5, label="Test")
        if scale == "log":
            axes[1].semilogy(ep, alldf["train_loss"], color="tab:blue", lw=2.5)
            axes[1].semilogy(ep, alldf["test_loss"], color="tab:red", lw=2.5)
        else:
            axes[1].plot(ep, alldf["train_loss"], color="tab:blue", lw=2.5)
            axes[1].plot(ep, alldf["test_loss"], color="tab:red", lw=2.5)
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Cross entropy loss")
        axes[0].legend(loc="lower right"); axes[0].set_ylim(0, 1.05)
        fig.suptitle("Grokking on modular addition")
        fig.tight_layout()
        fn = path / f"fig2_grokking_accuracy_loss_{scale}.png"
        fig.savefig(fn, dpi=150); plt.close(fig); out.append(fn)
    return out


# --------------------------------------------------------------------------- #
# Figure 3: Fourier norms of W_E and W_L
# --------------------------------------------------------------------------- #
def fig3_embedding_fourier(WE: np.ndarray, WL: np.ndarray, key_freqs: Sequence[int],
                           path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_e = WE.shape[0] // 2 + 1 if WE.shape[0] % 2 else WE.shape[0] // 2
    n_l = WL.shape[0] // 2 + 1
    norms_E = fourier_norms_1d(WE, axis=0)[:n_e]
    norms_L = fourier_norms_1d(WL, axis=0)[:n_l]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, norms, title in ((axes[0], norms_E, r"Embedding $W_E$"),
                             (axes[1], norms_L, r"Neuron-logit map $W_L$")):
        ax.bar(np.arange(norms.size), norms, color="steelblue")
        for k in key_freqs:
            if k < norms.size:
                ax.axvline(k, color="tab:red", ls=":", lw=1)
        ax.set_xlabel("Frequency"); ax.set_ylabel("Fourier norm")
        ax.set_title(title)
    fig.suptitle("Fourier norms of model weights (sparse in Fourier basis)")
    fig.tight_layout()
    fn = path.parent / f"{path.stem}.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    return fn


# --------------------------------------------------------------------------- #
# Figure 4: periodicity of attention, MLP neuron, and logits
# --------------------------------------------------------------------------- #
def fig4_periodicity(attn_grid: np.ndarray, neuron0: np.ndarray,
                     logits_2d_norm: np.ndarray, path: Path,
                     attn_title: str = "Attention =$\\to$a, head 0") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    im0 = axes[0].imshow(attn_grid, cmap="viridis", origin="lower")
    axes[0].set_title(attn_title); axes[0].set_xlabel("b"); axes[0].set_ylabel("a")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(neuron0, cmap="magma", origin="lower")
    axes[1].set_title("MLP neuron 0 activation"); axes[1].set_xlabel("b"); axes[1].set_ylabel("a")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(logits_2d_norm, cmap="inferno", origin="lower",
                         norm=matplotlib.colors.LogNorm(vmin=logits_2d_norm[logits_2d_norm > 0].min()))
    axes[2].set_title("Logits in 2D Fourier basis"); axes[2].set_xlabel("freq b"); axes[2].set_ylabel("freq a")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fn = path.parent / f"{path.stem}.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    return fn


# --------------------------------------------------------------------------- #
# Figure 5: neurons localised by frequency
# --------------------------------------------------------------------------- #
def fig5_neuron_localisation(counts: Dict, wl_heat: np.ndarray, path: Path,
                             cluster_label: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    freqs = [k for k in counts if k != "other/unfit"]
    vals = [counts[k] for k in freqs]
    axes[0].bar([str(k) for k in freqs], vals, color="steelblue")
    axes[0].set_xlabel("Best-fit frequency"); axes[0].set_ylabel("# neurons (FVE > 85%)")
    axes[0].set_title("MLP neurons localised by single frequency")
    im = axes[1].imshow(wl_heat, aspect="auto", cmap="inferno", origin="lower")
    axes[1].set_xlabel("Neurons in cluster"); axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"W$_L$ Fourier heatmap {cluster_label}")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fn = path.parent / f"{path.stem}.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    return fn


# --------------------------------------------------------------------------- #
# Figure 6: ablating individual frequencies
# --------------------------------------------------------------------------- #
def fig6_ablations(logits_grid: np.ndarray, labels: np.ndarray, key_freqs: Sequence[int],
                   path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    P = logits_grid.shape[0]
    freqs = np.arange(1, P // 2 + 1)
    # loss when ablating each single frequency k (its 4 products)
    excl_each = []
    for k in freqs:
        m = excluded_mask(P, [int(k)])
        F = np.fft.fft2(logits_grid, axes=(0, 1))
        F = F * m[:, :, None]
        abl = np.real(np.fft.ifft2(F, axes=(0, 1)))
        from .measures import log_softmax_loss
        excl_each.append(log_softmax_loss(abl, labels))
    excl_each = np.array(excl_each)
    # restricted loss = keep only key frequencies
    from .measures import log_softmax_loss
    rest = restricted_loss(logits_grid, labels, key_freqs)
    orig = log_softmax_loss(logits_grid, labels)
    log_chance = np.log(P)

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.bar(freqs, excl_each, color="steelblue", label="Loss ablating this frequency")
    ax.axhline(rest, color="tab:green", ls="--", lw=2, label="Restricted loss (keep key freqs)")
    ax.axhline(orig, color="tab:red", ls=":", lw=2, label="Original loss")
    ax.set_yscale("log")
    ax.set_ylim(min(orig * 0.5, 1e-8), max(log_chance * 1.5, excl_each.max() * 2))
    ax.set_xlabel("Ablated frequency"); ax.set_ylabel("Loss (log)")
    ax.legend()
    fig.tight_layout()
    fn = path.parent / f"{path.stem}.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    return fn


# --------------------------------------------------------------------------- #
# Figure 7: progress measures vs epoch (single run)
# --------------------------------------------------------------------------- #
def fig7_progress_measures(metrics: pd.DataFrame, progress: pd.DataFrame,
                           path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    ep = metrics["epoch"].values
    axes[0, 0].plot(ep, metrics["train_loss"], label="Train loss", color="tab:blue")
    axes[0, 0].plot(ep, metrics["test_loss"], label="Test loss", color="tab:red")
    if "excluded_loss_train" in progress:
        p = progress.dropna(subset=["excluded_loss_train"])
        axes[0, 0].plot(p["epoch"], p["excluded_loss_train"], label="Excluded loss (train)",
                        color="tab:green", lw=2)
    axes[0, 0].set_yscale("log"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Excluded loss rises while train/test loss are flat")

    axes[0, 1].plot(ep, metrics["test_loss"], label="Test loss", color="tab:red")
    if "restricted_loss_full" in progress:
        p = progress.dropna(subset=["restricted_loss_full"])
        axes[0, 1].plot(p["epoch"], p["restricted_loss_full"], label="Restricted loss (full)",
                        color="tab:green", lw=2)
    axes[0, 1].set_yscale("log"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Restricted loss drops before test loss")

    axes[1, 0].plot(progress["epoch"], progress["gini_WE"], label="Gini($W_E$)", color="tab:blue")
    axes[1, 0].plot(progress["epoch"], progress["gini_WL"], label="Gini($W_L$)", color="tab:red")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("Fourier sparsity (Gini) rises during cleanup")

    axes[1, 1].plot(progress["epoch"], progress["ssw"], color="tab:green")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("Sum of squared weights")
    axes[1, 1].set_title("Weight norm falls during circuit formation & cleanup")
    fig.suptitle("Progress measures over training")
    fig.tight_layout()
    fn = path.parent / f"{path.stem}.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    return fn


# --------------------------------------------------------------------------- #
# Figure 8: constructive interference of the key cosine waves
# --------------------------------------------------------------------------- #
def fig8_constructive_interference(p: int, freqs: Sequence[int], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, p, 1001)
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in freqs:
        ax.plot(x, np.cos(2 * np.pi * k / p * x), alpha=0.4, lw=1, label=f"k={k}")
    s = np.zeros_like(x)
    for k in freqs:
        s = s + np.cos(2 * np.pi * k / p * x)
    ax.plot(x, s / len(freqs), color="black", lw=2.5, label="Average of key freqs")
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(1, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("x  (mod p)"); ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Constructive interference at x=0 (mod p)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fn = path.parent / f"{path.stem}.png"
    fig.savefig(fn, dpi=150); plt.close(fig)
    return fn
