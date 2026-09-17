"""一次性探针（可删）：量化 5 个 seed 之间到底差在哪。

- 数据层面：train/test 划分是否随 seed 变（`make_modular_addition_data` 用 np.random.default_rng(seed)）
- 参数层面：各 seed 学出来的 W_E 关键频率是否一致
运行：python _seed_diff_probe.py
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grokking.data import make_modular_addition_data  # noqa: E402

P, FRAC = 113, 0.3
RUN = HERE.parent / "web" / "public" / "viz-data" / "mainline_p113_frac0.3_wd1_l1_d128_h4x32_m512_ep25000"


def load_bin(path: Path) -> dict[str, np.ndarray]:
    b = path.read_bytes()
    assert b[:4] == b"GKV1", path
    n_hdr = struct.unpack("<I", b[4:8])[0]
    hdr = json.loads(b[8:8 + n_hdr].decode("utf-8"))
    off, mats = 8 + n_hdr, {}
    for m in hdr["mats"]:
        size = int(np.prod(m["shape"]))
        mats[m["name"]] = np.frombuffer(b, dtype="<f4", count=size, offset=off).reshape(m["shape"])
        off += 4 * size
    return mats


def main() -> None:
    # ---------------------------------------------------------------- 数据划分
    masks = {}
    print("== 训练/测试划分（每个 seed 用 np.random.default_rng(seed) 抽 30%）==")
    for s in range(5):
        d = make_modular_addition_data(P, FRAC, s)
        idx = np.flatnonzero(d.train_mask)
        masks[s] = set(idx.tolist())
        print(f"seed {s}: train {len(idx)} 组，test {P * P - len(idx)} 组")
    for s in range(1, 5):
        inter = len(masks[0] & masks[s])
        print(f"  seed 0 ∩ seed {s} = {inter} 组共同训练样本（占训练集 {100 * inter / len(masks[0]):.1f}%）")

    # --------------------------------------------------------------- 学到的权重
    print("\n== 学到的 W_E 关键频率（对 token 轴做 FFT，看能量集中到哪几个 k）==")
    tops = {}
    for s in range(5):
        f = RUN / f"seed{s}" / "e025000.bin"
        if not f.exists():
            print(f"seed {s}: 缺 {f.name}")
            continue
        we = np.asarray(load_bin(f)["token_embed"][:P], dtype=np.float64)   # (P, d_model)
        spec = np.fft.fft(we, axis=0)
        n = 1 + P // 2
        energy = (np.abs(spec[:n]) ** 2).sum(axis=1)                       # 每个频率的总能量
        order = np.argsort(-energy)
        top = sorted(int(k) for k in order[:6])
        tops[s] = set(top)
        print(f"seed {s}: top-6 频率 {top}，能量占比 {100 * energy[order[:6]].sum() / energy.sum():.1f}%")

    ks = sorted(tops)
    if len(ks) > 1:
        base = tops[ks[0]]
        print()
        for s in ks[1:]:
            j = len(base & tops[s])
            print(f"  seed {ks[0]} 与 seed {s} 的 top-6 频率重合 {j}/6")
        union = set().union(*tops.values())
        inter = set.intersection(*tops.values())
        print(f"  5 个 seed 合起来出现的频率 {sorted(union)}")
        print(f"  5 个 seed 共有的频率 {sorted(inter)}")


if __name__ == "__main__":
    main()
