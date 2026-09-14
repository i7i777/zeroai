#!/usr/bin/env python3
"""Train one or more grokking experiments faithful to
Nanda et al. 2023 (arXiv:2301.05217).

Examples
--------
# mainline experiment, 1 seed, quick sanity on the full schedule
python run_experiment.py --preset mainline --seeds 0 --epochs 20000

# mainline, 5 seeds (paper Figure 2)
python run_experiment.py --preset mainline --seeds 0 1 2 3 4

# custom
python run_experiment.py --p 53 --wd 5.0 --frac 0.3 --seeds 0 --epochs 30000 \
    --tag p53_wd5

# re-run only the post-hoc progress-measure finalisation for an existing run
python run_experiment.py --finalize-only results/p113_frac0.3_wd1_.../seed0

Outputs (per seed dir under --out/results):
    config.json, metrics.csv, progress.csv, model_final.pt, data.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from grokking.config import GrokConfig, load_config
from grokking.train import run_experiment, finalize_progress_measures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["mainline"], default="mainline",
                    help="predefined experimental setup (default: mainline)")
    ap.add_argument("--p", type=int, default=113, help="modulus")
    ap.add_argument("--frac", type=float, default=0.3, help="training-data fraction")
    ap.add_argument("--wd", type=float, default=1.0, help="weight decay lambda")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-head", type=int, default=32)
    ap.add_argument("--n-mlp", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40_000, help="number of epochs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--grid-every", type=int, default=200,
                    help="store a logit grid every N epochs (for progress measures)")
    ap.add_argument("--no-grids", action="store_true",
                    help="do not store logit grids (skip progress measures)")
    ap.add_argument("--keep-grids", action="store_true",
                    help="keep grid files after finalisation")
    ap.add_argument("--viz-every", type=int, default=0,
                    help="export a visualizer weight snapshot every N epochs (0=off)")
    ap.add_argument("--viz-root", type=str, default="",
                    help="viz-data output directory (default: ../web/public/viz-data)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--tag", default="", help="tag prepended to run name")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch threads per process (default: all cores; use fewer "
                         "when running several seeds in parallel)")
    ap.add_argument("--finalize-only", type=str, default=None,
                    help="only finalise progress measures for an existing run dir")
    args = ap.parse_args()

    if args.threads is not None:
        import os
        os.environ["GROKKING_THREADS"] = str(args.threads)

    if args.finalize_only:
        finalize_progress_measures(Path(args.finalize_only), clean_grids=not args.keep_grids)
        print(f"finalised {args.finalize_only}")
        return

    for seed in args.seeds:
        cfg = GrokConfig(
            p=args.p, frac_train=args.frac, weight_decay=args.wd, lr=args.lr,
            d_model=args.d_model, n_heads=args.n_heads, d_head=args.d_head,
            n_mlp=args.n_mlp, n_epochs=args.epochs, seed=seed,
            grid_every=args.grid_every, save_grids=not args.no_grids,
            out_dir=args.out, tag=args.tag, viz_every=args.viz_every,
            viz_root=args.viz_root,
        )
        run_experiment(cfg, keep_grids=args.keep_grids)


if __name__ == "__main__":
    main()
