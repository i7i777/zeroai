"""Full-batch AdamW training loop plus per-epoch metrics and checkpoints.

Faithful to the paper's mainline procedure: one full-batch AdamW step per
epoch on a random 30% subset of the P*P pairs, 40,000 epochs, weight decay
lambda (AdamW decoupled) with lr=1e-3.

While training we additionally dump the full [P, P, P] logit grid every
`grid_every` epochs so that the expensive progress measures (restricted /
excluded loss, which need the *final* key frequencies) can be computed
post-hoc in `finalize_progress_measures`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import GrokConfig, load_config
from .data import make_modular_addition_data
from .measures import fourier_gini
from .model import OneLayerTransformer

MAX_THREADS = 16


def _set_threads() -> None:
    """Cap the number of torch threads.  The GROKKING_THREADS environment
    variable overrides the default (useful when running several seeds in
    parallel on one machine)."""
    n = min(MAX_THREADS, os.cpu_count() or 1)
    n = min(n, int(os.environ.get("GROKKING_THREADS", str(n))))
    torch.set_num_threads(n)


def evaluate(model: nn.Module, a: torch.Tensor, b: torch.Tensor,
             c: torch.Tensor, ce_loss: nn.Module) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(a, b)
        loss = ce_loss(logits, c)
        acc = (logits.argmax(-1) == c).float().mean()
    model.train()
    return {"loss": float(loss.item()), "acc": float(acc.item())}


@torch.no_grad()
def _logits_grid_for_epoch(model: OneLayerTransformer) -> np.ndarray:
    """Full [P, P, P] logit grid for the model in its current state."""
    lg = model.logits_grid()
    return lg.detach().cpu().numpy()


def run_experiment(cfg: GrokConfig, quiet: bool = False,
                   keep_grids: bool = False) -> Path:
    """Train one model according to cfg; returns the run directory."""
    _set_threads()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = Path(cfg.out_dir) / cfg.run_name() / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.json")

    # ---- data ----------------------------------------------------------- #
    data = make_modular_addition_data(cfg.p, cfg.frac_train, cfg.seed)
    np.savez(run_dir / "data.npz",
             a=data.a, b=data.b, c=data.c,
             train_mask=data.train_mask, test_mask=data.test_mask)
    a_tr = torch.from_numpy(data.a[data.train_mask])
    b_tr = torch.from_numpy(data.b[data.train_mask])
    c_tr = torch.from_numpy(data.c[data.train_mask])
    a_te = torch.from_numpy(data.a[data.test_mask])
    b_te = torch.from_numpy(data.b[data.test_mask])
    c_te = torch.from_numpy(data.c[data.test_mask])
    label_grid = data.label_grid()                 # [P, P]

    # ---- model / optimiser ---------------------------------------------- #
    model = OneLayerTransformer(
        p=cfg.p, d_model=cfg.d_model, n_heads=cfg.n_heads, d_head=cfg.d_head,
        n_mlp=cfg.n_mlp, n_layers=cfg.n_layers, init_std=cfg.init_std,
        seed=None,  # seeds already set above
    )
    model.train()
    ce_loss = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- loop ----------------------------------------------------------- #
    rows: List[Dict] = []
    progress_rows: List[Dict] = []
    grid_dir = run_dir / "grids"
    grid_dir.mkdir(exist_ok=True)
    n_params = sum(p.numel() for p in params)

    if not quiet:
        print(f"[run] p={cfg.p} frac={cfg.frac_train} wd={cfg.weight_decay} "
              f"lr={cfg.lr} n_epochs={cfg.n_epochs} seed={cfg.seed} "
              f"params={n_params} -> {run_dir}", flush=True)

    t0 = time.time()
    for epoch in range(1, cfg.n_epochs + 1):
        # one full-batch step
        opt.zero_grad(set_to_none=True)
        logits = model(a_tr, b_tr)
        loss = ce_loss(logits, c_tr)
        loss.backward()
        opt.step()

        if epoch % cfg.log_every == 0:
            tr = evaluate(model, a_tr, b_tr, c_tr, ce_loss)
            te = evaluate(model, a_te, b_te, c_te, ce_loss)
            rows.append({
                "epoch": epoch,
                "train_loss": tr["loss"], "train_acc": tr["acc"],
                "test_loss": te["loss"], "test_acc": te["acc"],
            })
            if not quiet and (epoch == 1 or epoch % max(cfg.log_every * 200, 1000) == 0):
                el = time.time() - t0
                print(f"  epoch {epoch:>6d}  train_loss={tr['loss']:.5f} "
                      f"train_acc={tr['acc']:.4f}  test_loss={te['loss']:.5f} "
                      f"test_acc={te['acc']:.4f}  ({el:.0f}s)", flush=True)

        # every `grid_every` epochs: dump logit grid + cheap weight measures
        if cfg.save_grids and (epoch % cfg.grid_every == 0 or epoch == cfg.n_epochs):
            grid = _logits_grid_for_epoch(model)
            np.save(grid_dir / f"e{epoch:06d}.npy", grid.astype(np.float16))
            wE = model.W_E.detach().cpu().numpy()
            wL = model.W_L.detach().cpu().numpy()
            ssw = float(sum((p.detach() ** 2).sum().item() for p in params))
            progress_rows.append({
                "epoch": epoch,
                "gini_WE": fourier_gini(wE, axis=0),
                "gini_WL": fourier_gini(wL, axis=0),
                "ssw": ssw,
            })
            del grid

        # every `viz_every` epochs: export a weight snapshot for the web visualizer
        if cfg.viz_every and (epoch == 1 or epoch % cfg.viz_every == 0
                              or epoch == cfg.n_epochs):
            try:
                from .vizexport import export_snapshot, viz_root
                root = Path(cfg.viz_root) if cfg.viz_root else viz_root()
                export_snapshot(model, cfg, epoch, root=root)
            except Exception as exc:  # pragma: no cover - do not kill training
                print(f"[viz] snapshot export failed at epoch {epoch}: {exc}", flush=True)

        # periodic flush so an interrupted run still keeps most metrics
        if epoch % 1000 == 0:
            pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
            pd.DataFrame(progress_rows).to_csv(run_dir / "progress_raw.csv", index=False)

    # ---- save ----------------------------------------------------------- #
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(progress_rows).to_csv(run_dir / "progress_raw.csv", index=False)
    torch.save({"model_state": model.state_dict(), "epoch": cfg.n_epochs,
                "config": cfg.to_dict()}, run_dir / "model_final.pt")

    if not quiet:
        print(f"[run] done in {time.time()-t0:.0f}s. results in {run_dir}", flush=True)

    # ---- post-hoc progress measures ------------------------------------ #
    if cfg.save_grids:
        finalize_progress_measures(run_dir, clean_grids=not keep_grids, quiet=quiet)

    return run_dir


def finalize_progress_measures(run_dir: Path, clean_grids: bool = True,
                               quiet: bool = False) -> Path:
    """Compute restricted / excluded loss for every stored logit grid using the
    key frequencies detected from the *final* model, then merge everything into
    progress.csv.  Optionally deletes the grid files afterwards."""
    from .measures import excluded_loss, restricted_loss
    from .analysis import detect_key_freqs_of_model

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.json")
    grid_dir = run_dir / "grids"

    data = make_modular_addition_data(cfg.p, cfg.frac_train, cfg.seed)
    label_grid = data.label_grid()                        # [P, P]
    flat_train_idx = np.nonzero(data.train_mask)[0]
    flat_test_idx = np.nonzero(data.test_mask)[0]

    # final model -> key frequencies
    ckpt = torch.load(run_dir / "model_final.pt", map_location="cpu")
    model = OneLayerTransformer(p=cfg.p, d_model=cfg.d_model, n_heads=cfg.n_heads,
                                d_head=cfg.d_head, n_mlp=cfg.n_mlp,
                                n_layers=cfg.n_layers, init_std=cfg.init_std)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    key_freqs = detect_key_freqs_of_model(model)
    if not quiet:
        print(f"[finalize] {run_dir.name}: key frequencies = {key_freqs}", flush=True)

    grid_files = sorted(grid_dir.glob("e*.npy"), key=lambda p: int(p.stem[1:]))
    prog = pd.DataFrame(columns=["epoch", "gini_WE", "gini_WL", "ssw"])
    raw = run_dir / "progress_raw.csv"
    if raw.exists():
        prog = pd.read_csv(raw)

    rest_full, excl_train, excl_full = [], [], []
    for f in grid_files:
        grid = np.load(f).astype(np.float32)             # [P, P, P]
        rest_full.append(restricted_loss(grid, label_grid, key_freqs))
        excl_train.append(excluded_loss(grid, label_grid, key_freqs, flat_train_idx))
        excl_full.append(excluded_loss(grid, label_grid, key_freqs))
        del grid

    rows = []
    for i, f in enumerate(grid_files):
        ep = int(f.stem[1:])
        rows.append({"epoch": ep, "restricted_loss_full": rest_full[i],
                     "excluded_loss_train": excl_train[i],
                     "excluded_loss_full": excl_full[i]})
    res = pd.DataFrame(rows)
    if len(prog):
        prog = prog[prog["epoch"].isin(res["epoch"])].merge(
            res, on="epoch", how="outer").sort_values("epoch")
    else:
        prog = res.sort_values("epoch")
    prog.to_csv(run_dir / "progress.csv", index=False)

    # keep raw + merged; optionally drop grids
    if clean_grids and grid_dir.exists():
        for f in grid_dir.glob("*.npy"):
            f.unlink()
        try:
            grid_dir.rmdir()
        except OSError:
            pass

    # stash the key frequencies for the analysis stage
    (run_dir / "key_freqs.json").write_text(json.dumps(list(map(int, key_freqs))))
    return run_dir
