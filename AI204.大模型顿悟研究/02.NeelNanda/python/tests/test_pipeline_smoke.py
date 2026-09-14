#!/usr/bin/env python3
"""Lightweight smoke test of the analysis & plotting pipeline on a SMALL model,
so that we can catch runtime errors without waiting for a full grokked run.

python tests/test_pipeline_smoke.py
"""
import os
os.environ.setdefault("GROKKING_THREADS", "1")
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from grokking.data import make_modular_addition_data
from grokking.model import OneLayerTransformer
from grokking.measures import (gini_coefficient, detect_key_freqs,
                               restricted_loss, excluded_loss,
                               restricted_mask, excluded_mask,
                               fourier_norms_matrix)
from grokking.analysis import (attention_pattern_grid, mlp_activations_grid,
                               neuron_frequency_fit, wl_fourier_heatmap,
                               wl_columns_for_cluster, detect_key_freqs_of_model)
from grokking.plots import (fig3_embedding_fourier, fig4_periodicity,
                            fig5_neuron_localisation, fig6_ablations,
                            fig8_constructive_interference,
                            fig2_grokking_curves, fig7_progress_measures)
from grokking.fourier import logits_fourier_norm
import pandas as pd

OUT = HERE / "_smoke_out"
OUT.mkdir(exist_ok=True)
p = 23

# ---- data / model ------------------------------------------------------ #
data = make_modular_addition_data(p, 0.3, 0)
model = OneLayerTransformer(p=p, d_model=32, n_heads=2, d_head=16,
                            n_mlp=64, init_std=0.02)
# give it a few random training steps so logits are non-trivial
a = torch.from_numpy(data.a[data.train_mask])
b = torch.from_numpy(data.b[data.train_mask])
c = torch.from_numpy(data.c[data.train_mask])
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.5)
for _ in range(120):
    opt.zero_grad()
    lg = model(a, b)
    loss = torch.nn.functional.cross_entropy(lg, c)
    loss.backward()
    opt.step()

# ---- matrices ---------------------------------------------------------- #
WE = model.W_E.numpy()
WL = model.W_L.numpy()
key = detect_key_freqs_of_model(model, min_frac=0.1)
print("key freqs:", key)
fig3_embedding_fourier(WE, WL, key, OUT / "t_fig3.png")

grid = model.logits_grid().numpy()
labels = data.label_grid()
print("grid", grid.shape, "restricted_mask", restricted_mask(p, key).sum(),
      "excluded_mask", excluded_mask(p, key).sum())
print("restricted loss:", restricted_loss(grid, labels, key))
print("excluded loss:", excluded_loss(grid, labels, key,
                                      np.nonzero(data.train_mask)[0]))

# analysis
attn0 = attention_pattern_grid(model, 0)
acts = mlp_activations_grid(model)
print("attn/acts", attn0.shape, acts.shape)
fit = neuron_frequency_fit(acts, freqs=key)
print("neuron fit counts:", fit["counts"])
counts = fit["counts"]
big_k = max([k for k in counts if k != "other/unfit"], key=lambda k: counts[k],
            default=key[0] if key else 1)
idx = np.nonzero((fit["best_freq"] == big_k))[0]
heat = wl_fourier_heatmap(wl_columns_for_cluster(WL, idx))
fig5_neuron_localisation(counts, heat, OUT / "t_fig5.png",
                         cluster_label=f"(freq {big_k}, {len(idx)} n)")

L2 = logits_fourier_norm(grid)
fig4_periodicity(attn0, acts[..., 0], L2, OUT / "t_fig4.png")
fig6_ablations(grid, labels, key, OUT / "t_fig6.png")
fig8_constructive_interference(p, key, OUT / "t_fig8.png")

# fig7 with fabricated progress/metrics
m = pd.DataFrame({"epoch": np.arange(1, 2001, 50),
                  "train_loss": np.linspace(3, 1e-2, 40),
                  "train_acc": np.linspace(0.1, 1.0, 40),
                  "test_loss": np.linspace(3, 2, 40),
                  "test_acc": np.linspace(0.1, 0.3, 40)})
pr = pd.DataFrame({"epoch": np.arange(1, 2001, 50),
                   "gini_WE": np.linspace(0.2, 0.9, 40),
                   "gini_WL": np.linspace(0.2, 0.9, 40),
                   "ssw": np.linspace(50, 5, 40),
                   "restricted_loss_full": np.linspace(3, 0.1, 40),
                   "excluded_loss_train": np.linspace(0.5, 2, 40)})
fig7_progress_measures(m, pr, OUT / "t_fig7.png")

# fig2 fabricated
m2 = m.copy(); m2["epoch"] = m2["epoch"] * 2
fig2_grokking_curves([m, m2], OUT / "t_fig2")

print("smoke OK ->", OUT)
