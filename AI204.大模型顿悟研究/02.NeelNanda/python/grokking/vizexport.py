"""Export model weight snapshots for the web visualizer.

The web visualizer (`web/public/model.html`) lets the user load the weights of
a chosen seed at a chosen training stage (epoch) and inspect them (heatmaps,
Fourier norms ...).  For that we store compact binary snapshots of the weight
matrices.

Snapshot format (one `.bin` per epoch):
    bytes  0..3     magic  "GKV1"
    bytes  4..7     uint32  length of the JSON header (little-endian)
    bytes  8..8+L   JSON header, e.g.
                    {"run":..,"seed":..,"epoch":..,"config":{..},
                     "mats":[{"name":"token_embed","shape":[114,128],
                              "start":0,"numel":14592}, ...]}
    then           float32 (little-endian) values of all matrices, concatenated
                   in the order given by `mats`.

In addition an `index.json` is kept per (run, seed) listing the available
epochs and a down-sampled training curve, and one global
`web/public/viz-data/index.json` aggregates everything for the front end.

`export_split()` 另外给每个 seed 写一份 `split.json`：整张 P×P 训练/测试划分的**位图**
（base64），供前端画「模加法表」——蓝 = 训练集、红 = 留出（测试）集。
它只用 `grokking.data.make_modular_addition_data`（与训练完全同一段代码）重算划分，
不读快照、不重训，所以随时可以补写。
"""
from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .config import GrokConfig, load_config
from .model import OneLayerTransformer

MAGIC = b"GKV1"
CURVE_DOWNSAMPLE = 25          # keep every Nth epoch in the curve
VIZ_ROOT_DEFAULT = None        # resolved relative to the repo (see below)


def _repo_root() -> Path:
    # python/grokking/vizexport.py -> python -> 02.NeelNanda
    return Path(__file__).resolve().parents[2]


def viz_root() -> Path:
    return _repo_root() / "web" / "public" / "viz-data"


# --------------------------------------------------------------------------- #
# matrix extraction
# --------------------------------------------------------------------------- #
def matrices_of_model(model: OneLayerTransformer) -> List[Tuple[str, np.ndarray]]:
    def np_(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    mats: List[Tuple[str, np.ndarray]] = []
    mats.append(("token_embed", np_(model.token_embed.weight)))   # [p+1, d]
    mats.append(("pos_embed", np_(model.pos_embed)))              # [3, d]
    for i, h in enumerate(model.heads):
        mats.append((f"head{i}.W_Q", np_(h.W_Q.weight)))          # [d, d_head]
        mats.append((f"head{i}.W_K", np_(h.W_K.weight)))
        mats.append((f"head{i}.W_V", np_(h.W_V.weight)))
        mats.append((f"head{i}.W_O", np_(h.W_O.weight)))          # [d_head, d]
    mats.append(("mlp_in.w", np_(model.mlp_in.weight)))           # [n_mlp, d]
    mats.append(("mlp_in.b", np_(model.mlp_in.bias)))             # [n_mlp]
    mats.append(("mlp_out.w", np_(model.mlp_out.weight)))         # [d, n_mlp]
    mats.append(("mlp_out.b", np_(model.mlp_out.bias)))           # [d]
    mats.append(("unembed", np_(model.unembed.weight)))           # [p, d]
    return mats


def config_brief(cfg: GrokConfig) -> Dict:
    return {
        "p": cfg.p, "d_model": cfg.d_model, "n_heads": cfg.n_heads,
        "d_head": cfg.d_head, "n_mlp": cfg.n_mlp, "n_layers": cfg.n_layers,
        "weight_decay": cfg.weight_decay, "lr": cfg.lr,
        "frac_train": cfg.frac_train, "n_epochs": cfg.n_epochs,
    }


def build_snapshot(model: OneLayerTransformer, run: str, seed: int, epoch: int,
                   cfg: GrokConfig) -> bytes:
    mats = matrices_of_model(model)
    total = sum(m.size for _, m in mats)
    start = 0
    header_mats = []
    for name, m in mats:
        header_mats.append({"name": name, "shape": list(m.shape),
                            "start": start, "numel": int(m.size)})
        start += int(m.size)
    header = {"run": run, "seed": seed, "epoch": epoch,
              "config": config_brief(cfg), "mats": header_mats}
    hbytes = json.dumps(header).encode("utf-8")
    body = np.concatenate([m.reshape(-1).astype(np.float32) for _, m in mats])
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", len(hbytes))
    out += hbytes
    out += body.tobytes()
    return bytes(out)


# --------------------------------------------------------------------------- #
# 数据划分（供前端画“模加法表”）
# --------------------------------------------------------------------------- #
def export_split(cfg: Dict, seed: int, seed_dir: Path) -> Path:
    """写出该 seed 的完整 P×P 训练/测试划分（位图，base64）→ `split.json`。

    用的是与训练**完全相同**的那段 numpy 代码（`p*grokking.data.make_modular_addition_data`：
    `rng.permutation` 后前 `frac_train` 部分当训练集），所以不重训、不读快照也能量出来，
    前端「模加法表」就是拿它把 (a, b) 一格一格涂成蓝（训练）/ 红（留出）。
    """
    import base64

    from .data import make_modular_addition_data

    p = int(cfg["p"])
    frac = float(cfg.get("frac_train", 0.3))
    data = make_modular_addition_data(p, frac, int(seed))
    # 注意：np.packbits 默认是 MSB 先行，而前端按 LSB 先行解（第 j 字节的第 i 位 = mask[j*8+i]），
    # 这里必须显式指定 little，否则每 8 格里会整体错位（1 的个数不变，肉眼看不出，但格子是对错的）。
    packed = np.packbits(data.train_mask, bitorder="little")
    out = {
        "kind": "numpy split (与训练互斥)",
        "p": p, "frac_train": frac, "seed": int(seed),
        "n": int(data.a.size),
        "n_train": int(data.train_mask.sum()),
        "n_test": int(data.test_mask.sum()),
        "encoding": "bits-lsb-first-base64",
        "train": base64.b64encode(packed.tobytes()).decode("ascii"),
    }
    path = seed_dir / "split.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# index book-keeping
# --------------------------------------------------------------------------- #
def _curve(run_dir: Path) -> Dict:
    mf = run_dir / "metrics.csv"
    if not mf.exists():
        return {}
    df = pd.read_csv(mf)
    df = df.iloc[::CURVE_DOWNSAMPLE]
    cols = [c for c in ("epoch", "train_loss", "train_acc", "test_loss", "test_acc") if c in df]
    if not cols:
        return {}
    rec = {c: [float(x) for x in df[c].tolist()] for c in cols}
    pf = run_dir / "progress.csv"
    if pf.exists():
        p = pd.read_csv(pf)
        for c in ("restricted_loss_full", "excluded_loss_train", "gini_WE", "gini_WL"):
            if c in p:
                rec[c] = [float(x) for x in p[c].tolist()]
        rec["progress_epoch"] = [float(x) for x in p["epoch"].tolist()]
    return rec


def _write_seed_index(seed_dir: Path, epochs: List[int]) -> Dict:
    idx = json.loads((seed_dir / "index.json").read_text(encoding="utf-8")) \
        if (seed_dir / "index.json").exists() else {}
    idx["epochs"] = sorted(set(idx.get("epochs", []) + [int(e) for e in epochs]))
    (seed_dir / "index.json").write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    return idx


def _write_global_index(root: Path) -> None:
    runs: Dict[str, Dict] = {}
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        seeds = {}
        for seed_dir in sorted(run_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            idxp = seed_dir / "index.json"
            if idxp.exists():
                idx = json.loads(idxp.read_text(encoding="utf-8"))
                seeds[idx.get("seed", seed_dir.name)] = idx
        if seeds:
            runs[run_dir.name] = {"seeds": seeds}
    (root / "index.json").write_text(
        json.dumps({"runs": runs}, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# high-level exports
# --------------------------------------------------------------------------- #
def export_snapshot(model: OneLayerTransformer, cfg: GrokConfig, epoch: int,
                    run: Optional[str] = None, seed: Optional[int] = None,
                    root: Optional[Path] = None,
                    curve: Optional[Dict] = None) -> Path:
    """Write one snapshot into <root>/<run>/<seed>/e{epoch:06d}.bin and update
    the per-seed index.  Returns the written path."""
    root = Path(root) if root else viz_root()
    run = run or cfg.run_name()
    seed = seed if seed is not None else cfg.seed
    seed_dir = root / run / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    data = build_snapshot(model, run, seed, epoch, cfg)
    fname = f"e{int(epoch):06d}.bin"
    (seed_dir / fname).write_bytes(data)

    idxp = seed_dir / "index.json"
    idx = json.loads(idxp.read_text(encoding="utf-8")) if idxp.exists() else {}
    idx.update({"run": run, "seed": seed, "config": config_brief(cfg)})
    if curve:
        idx["curve"] = curve
    idx.setdefault("epochs", [])
    if int(epoch) not in idx["epochs"]:
        idx["epochs"].append(int(epoch))
        idx["epochs"].sort()
    idxp.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
    _write_global_index(root)
    return seed_dir / fname


def export_final_of_run(run_dir: Path, root: Optional[Path] = None) -> List[Path]:
    """Export the final model of every seed under a results/<run> directory."""
    run_dir = Path(run_dir)
    root = Path(root) if root else viz_root()
    written: List[Path] = []
    seeds = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed"))
    if not seeds and (run_dir / "model_final.pt").exists():
        seeds = [run_dir]          # run_dir is itself a seed dir
    for sd in seeds:
        ck = sd / "model_final.pt"
        cfgp = sd / "config.json"
        if not ck.exists() or not cfgp.exists():
            print(f"[viz] skip {sd}: missing model_final.pt/config.json", flush=True)
            continue
        cfg = load_config(cfgp)
        ckpt = torch.load(ck, map_location="cpu")
        model = OneLayerTransformer(p=cfg.p, d_model=cfg.d_model, n_heads=cfg.n_heads,
                                    d_head=cfg.d_head, n_mlp=cfg.n_mlp,
                                    n_layers=cfg.n_layers, init_std=cfg.init_std)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        seed_no = int(sd.name.replace("seed", ""))
        curve = _curve(sd)
        p = export_snapshot(model, cfg, cfg.n_epochs, run=cfg.run_name(),
                            seed=seed_no, root=root, curve=curve)
        written.append(p)
        print(f"[viz] exported final {sd.name} -> {p}", flush=True)
    return written


def export_all_finals(results_root: Path, root: Optional[Path] = None) -> None:
    results_root = Path(results_root)
    for run_dir in sorted(results_root.iterdir()):
        if (run_dir / "config.json").exists():
            export_final_of_run(run_dir, root=root)


if __name__ == "__main__":
    # 用法: python -m grokking.vizexport --root <web/public/viz-data> [results dirs...]
    import sys
    args = sys.argv[1:]
    root = None
    if "--root" in args:
        i = args.index("--root")
        root = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    for r in args:
        export_final_of_run(Path(r), root=root)
