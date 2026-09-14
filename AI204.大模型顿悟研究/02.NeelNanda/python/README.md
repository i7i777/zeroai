# Reproduction of *Progress measures for grokking via mechanistic interpretability*

Faithful PyTorch re-implementation of **Nanda, Chan, Lieberum, Smith & Steinhardt
(ICLR 2023)**, arXiv:2301.05217 — the paper stored in `../docs/`.  Everything
runs on **CPU** with a small model (~227k parameters).

## What is reproduced

* **Grokking on modular addition** (`a + b mod P`): a one-layer ReLU transformer
  first memorises the training pairs, then — thanks to weight decay — suddenly
  generalises (the "grokking" curve, paper Figure 2).
* **The reverse-engineered algorithm** ("Fourier multiplication"): the weights
  and activations are sparse in the Fourier basis; the model combines sines and
  cosines at a handful of *key frequencies* (paper Figures 3–6).
* **Progress measures** that move continuously *before* grokking happens:
  restricted / excluded loss, the Gini coefficient of the Fourier norms of
  $W_E$ and $W_L$, and the sum of squared weights — exposing the three phases
  *memorisation, circuit formation, cleanup* (paper Figure 7).
* **Constructive interference** of the key cosine waves (paper Figure 8).

## Layout

```
python/
├── grokking/                  # the reproduction package
│   ├── config.py              # hyper-parameters (mainline presets & variants)
│   ├── data.py                # modular-addition dataset + train/test split
│   ├── model.py               # one-layer ReLU transformer (paper Appendix A)
│   ├── fourier.py             # DFT helpers
│   ├── measures.py            # restricted/excluded loss, Gini, key-freq masks
│   ├── analysis.py            # post-hoc reverse engineering of a trained model
│   ├── train.py               # full-batch AdamW training + checkpointing
│   └── plots.py               # matplotlib rendering of Figures 2–8
├── run_experiment.py          # CLI: train experiments
├── make_figures.py            # CLI: produce the paper's figures from runs
├── requirements.txt
├── results/                   # per-run outputs (metrics.csv, model_final.pt, ...)
└── figures/                   # generated figures
```

## Quick start

```bash
# 1. train the mainline experiment (paper Section 3): 30% data, wd=1, lr=1e-3
python run_experiment.py --seeds 0 1 2 3 4 --epochs 40000 --out results

# 2. reproduce Figures 2–8 (detailed analysis on seed 0, curves over all seeds)
python make_figures.py --run-dir results/mainline_p113_frac0.3_wd1_l1_d128_h4x32_m512_ep40000 \
    --analysis-seeds 0 --out figures
```

On a CPU-only machine 40 000 epochs take a while (~0.1 s/epoch ⇒ ~1.5 h/seed).
A shorter run (e.g. `--epochs 20000`) still shows the full grokking curve and
all three phases.

### Important settings (paper mainline)

| setting | value |
|---|---|
| modulus P | 113 |
| training data | 30 % of all 113×113 pairs |
| architecture | 1-layer ReLU transformer, d_model=128, 4 heads × d_head=32, MLP 512 |
| regularisation | none (no LayerNorm, no dropout); embed/unembed untied |
| optimiser | full-batch **AdamW**, lr = 1e-3, **weight decay λ = 1.0** |
| epochs | 40 000 (paper) |
| input | tokens `a b =`; loss read off the `=` token |

## Output of a single run (`results/<run>/seed<i>/`)

| file | contents |
|---|---|
| `metrics.csv` | per-epoch train/test loss & accuracy |
| `progress_raw.csv` | per-`grid_every`-epoch Gini($W_E$), Gini($W_L$), Σw² |
| `progress.csv` | merged with post-hoc restricted/excluded loss |
| `model_final.pt` | final checkpoint |
| `data.npz` | dataset (a,b,c, masks) |
| `config.json` | hyper-parameters |
| `key_freqs.json` | key frequencies detected from the final model |

## Mapping of the figures

| paper figure | content | produced by |
|---|---|---|
| Fig. 2 | grokking curves (5 seeds) | `make_figures.py` → `fig2_grokking_accuracy_loss_{linear,log}.png` |
| Fig. 3 | Fourier norms of $W_E$ and $W_L$ | `fig3_fourier_norms_seed0.png` |
| Fig. 4 | attention / MLP-neuron periodicity + logits in Fourier basis | `fig4_periodicity_seed0.png` |
| Fig. 5 | neurons localised by frequency | `fig5_neurons_seed0.png` |
| Fig. 6 | ablating frequencies in Fourier space | `fig6_ablations_seed0.png` |
| Fig. 7 | progress measures & three phases | `fig7_progress_seed0.png` |
| Fig. 8 | constructive interference | `fig8_interference_seed0.png` |
| Fig. D.1 (app.) | no grokking without weight decay | run with `--wd 0` and plot Fig-2-style |
| Fig. 27 (app.) | effect of λ = 0.3 / 3.0 | run with `--wd 0.3` / `--wd 3.0` |

## Reproducing the appendix / ablation experiments

```bash
# without weight decay (no grokking, paper Figure 26)
python run_experiment.py --wd 0.0 --epochs 15000 --seeds 0 1 2 --tag wd0 --out results

# stronger / weaker weight decay (paper Figure 27)
python run_experiment.py --wd 0.3 --epochs 30000 --seeds 0 1 2 --tag wd0.3 --out results
python run_experiment.py --wd 3.0 --epochs 40000 --seeds 0 1 2 --tag wd3 --out results

# smaller modulus needs stronger decay (paper Figure 24: P=53, λ=5)
python run_experiment.py --p 53 --wd 5.0 --epochs 30000 --seeds 0 1 2 --tag p53_wd5 --out results

# more / less training data (paper Figures 20–21)
python run_experiment.py --frac 0.5 --epochs 10000 --seeds 0 --tag frac0.5 --out results
```

Every such run lands in its own folder and can be plotted with
`make_figures.py --run-dir results/<run> ...`.

## Methodological notes & caveats

* Progress measures (restricted / excluded loss) need the *final* key
  frequencies, so they are computed **post-hoc**: during training a full
  `P×P×P` logit grid is dumped every `--grid-every` epochs and consumed by
  `finalize_progress_measures()` once the run finishes (grids are deleted
  afterwards unless `--keep-grids`).
* `restricted loss` is reported on the full input grid (a generalising circuit
  should score well on all pairs); `excluded loss` is reported on the training
  pairs, matching the paper.
* Key frequencies are detected per trained model (DFT of $W_L$, "nontrivial
  coefficients"), which — as in the paper — varies by seed.
* Detailed reverse-engineering figures (3–8) are generated for the seeds given
  to `--analysis-seeds` (default seed 0); the grokking-curve figure always uses
  every available seed.
