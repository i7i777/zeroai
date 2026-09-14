#!/usr/bin/env python3
"""Reproduce the paper's main figures from trained runs.

python make_figures.py --run-dir results/<run_name> [--analysis-seeds 0] [--out figures]

For a run directory that contains seed sub-folders (seed0, seed1, ...):
  * Figure 2  (grokking curves) uses ALL seeds.
  * Figures 3-8 (mechanistic analysis & progress measures) are generated for
    the seeds passed to --analysis-seeds (default: seed 0 only).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from grokking.analysis import (attention_pattern_grid, attention_patterns_all_heads,
                               mlp_activations_grid, neuron_frequency_fit,
                               wl_columns_for_cluster, wl_fourier_heatmap,
                               detect_key_freqs_of_model)
from grokking.config import load_config
from grokking.data import make_modular_addition_data
from grokking.fourier import logits_fourier_norm
from grokking.model import OneLayerTransformer
from grokking.plots import (fig2_grokking_curves, fig3_embedding_fourier,
                            fig4_periodicity, fig5_neuron_localisation,
                            fig6_ablations, fig7_progress_measures,
                            fig8_constructive_interference)
from grokking.train import _set_threads


def _load_model(run_dir: Path):
    cfg = load_config(run_dir / "config.json")
    ckpt = torch.load(run_dir / "model_final.pt", map_location="cpu")
    model = OneLayerTransformer(p=cfg.p, d_model=cfg.d_model, n_heads=cfg.n_heads,
                                d_head=cfg.d_head, n_mlp=cfg.n_mlp,
                                n_layers=cfg.n_layers, init_std=cfg.init_std)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return cfg, model


def _key_freqs(run_dir: Path, model) -> list:
    kf = run_dir / "key_freqs.json"
    if kf.exists():
        return json.loads(kf.read_text())
    return detect_key_freqs_of_model(model)


def _figure_dir(out_root: Path, run_name: str) -> Path:
    d = out_root / run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def analyse_seed(run_dir: Path, out_dir: Path, cfg, model, key_freqs,
                 do_fig5: bool = True) -> None:
    name = f"seed{Path(run_dir).name.replace('seed','')}"
    fdir = _figure_dir(out_dir, f"{cfg.run_name()}")
    p = cfg.p

    # ---- Figure 3 ------------------------------------------------------ #
    WE = model.W_E.detach().cpu().numpy()     # [p, d]
    WL = model.W_L.detach().cpu().numpy()     # [p, n_mlp]
    fig3_embedding_fourier(WE, WL, key_freqs,
                           fdir / f"fig3_fourier_norms_{name}.png")

    # ---- logits grid --------------------------------------------------- #
    lg = model.logits_grid().detach().cpu().numpy()      # [p, p, p]
    labels = make_modular_addition_data(cfg.p, cfg.frac_train, cfg.seed).label_grid()

    # ---- Figure 4 ------------------------------------------------------ #
    attn0 = attention_pattern_grid(model, head=0)
    acts = mlp_activations_grid(model)                   # [p, p, n]
    L2 = logits_fourier_norm(lg)
    fig4_periodicity(attn0, acts[..., 0], L2,
                     fdir / f"fig4_periodicity_{name}.png")

    # ---- Figure 5 (neuron localisation) -------------------------------- #
    if do_fig5:
        fit = neuron_frequency_fit(acts, freqs=key_freqs)
        # largest cluster -> W_L Fourier heatmap
        counts = fit["counts"]
        best_freq = fit["best_freq"]
        big_k = max([k for k in counts if k != "other/unfit"],
                    key=lambda k: counts[k], default=key_freqs[0] if key_freqs else 1)
        idx = np.nonzero((best_freq == big_k))[0]
        wl_heat = wl_fourier_heatmap(wl_columns_for_cluster(WL, idx))
        fig5_neuron_localisation(counts, wl_heat,
                                 fdir / f"fig5_neurons_{name}.png",
                                 cluster_label=f"(freq {big_k}, {len(idx)} neurons)")
    del acts

    # ---- Figure 6 ------------------------------------------------------ #
    fig6_ablations(lg, labels, key_freqs, fdir / f"fig6_ablations_{name}.png")

    # ---- Figure 8 ------------------------------------------------------ #
    fig8_constructive_interference(p, key_freqs, fdir / f"fig8_interference_{name}.png")

    # ---- Figure 7 (progress measures) ---------------------------------- #
    metrics = pd.read_csv(run_dir / "metrics.csv")
    progress = pd.read_csv(run_dir / "progress.csv")
    fig7_progress_measures(metrics, progress, fdir / f"fig7_progress_{name}.png")


def main() -> None:
    _set_threads()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="path to results/<run_name> containing seed sub-folders")
    ap.add_argument("--analysis-seeds", type=int, nargs="+", default=[0],
                    help="which seeds get the detailed mechanistic figures")
    ap.add_argument("--out", default="figures", help="output directory")
    ap.add_argument("--no-fig5", action="store_true", help="skip slow neuron fitting")
    ap.add_argument("--fig2-only", action="store_true",
                    help="only produce the grokking-curve figure (works on partial "
                         "or unfinished runs), then exit")
    args = ap.parse_args()

    root = Path(args.run_dir)
    seed_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed")],
                       key=lambda d: int(d.name.replace("seed", "")))
    if not seed_dirs:
        # maybe run-dir is itself a seed dir
        if (root / "model_final.pt").exists():
            seed_dirs = [root]
        else:
            raise SystemExit(f"no seed dirs found under {root}")

    # Figure 2 uses all seeds ------------------------------------------------
    metrics_all = []
    for sd in seed_dirs:
        m = pd.read_csv(sd / "metrics.csv")
        metrics_all.append(m)
    run_name = load_config(seed_dirs[0] / "config.json").run_name()
    fdir = _figure_dir(Path(args.out), run_name)
    fig2_grokking_curves(metrics_all, fdir)
    print(f"[fig2] saved under {fdir}")
    if args.fig2_only:
        print("[fig2-only] done")
        return

    # Mechanistic figures for the requested seeds -----------------------------
    for sd in seed_dirs:
        seed_no = int(sd.name.replace("seed", ""))
        if seed_no not in args.analysis_seeds:
            continue
        cfg, model = _load_model(sd)
        key_freqs = _key_freqs(sd, model)
        print(f"[mech] seed {seed_no}: key frequencies = {key_freqs}", flush=True)
        analyse_seed(sd, Path(args.out), cfg, model, key_freqs,
                     do_fig5=not args.no_fig5)


if __name__ == "__main__":
    main()
