"""一次性探针（可删）：5 个 seed 的“学到什么”在**输出端**（logits）是否一致。

对每个 seed 载入最终快照的权重，跑一次 a=0、b=0..P-1 的前向，
对 logits 沿 b 做 DFT —— 谱峰就是该网络真正用到的频率（论文 4.1 的做法）。
运行：python _seed_diff_probe2.py
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grokking.model import OneLayerTransformer  # noqa: E402

P = 113
RUN = HERE.parent / "web" / "public" / "viz-data" / "mainline_p113_frac0.3_wd1_l1_d128_h4x32_m512_ep25000"


def load_bin(path: Path) -> dict[str, np.ndarray]:
    b = path.read_bytes()
    n_hdr = struct.unpack("<I", b[4:8])[0]
    hdr = json.loads(b[8:8 + n_hdr].decode("utf-8"))
    off, mats = 8 + n_hdr, {}
    for m in hdr["mats"]:
        size = int(np.prod(m["shape"]))
        mats[m["name"]] = np.frombuffer(b, dtype="<f4", count=size, offset=off).reshape(m["shape"])
        off += 4 * size
    return hdr, mats


def build(mats: dict[str, np.ndarray], cfg: dict) -> OneLayerTransformer:
    m = OneLayerTransformer(p=cfg["p"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                            d_head=cfg["d_head"], n_mlp=cfg["n_mlp"], n_layers=cfg["n_layers"])
    sd = {}
    for k, v in mats.items():
        t = torch.tensor(np.asarray(v), dtype=torch.float32)
        if k == "token_embed":
            sd["token_embed.weight"] = t
        elif k == "pos_embed":
            sd["pos_embed"] = t
        elif k == "unembed":
            sd["unembed.weight"] = t
        elif k.startswith("head"):
            h, _, w = k.partition(".")
            sd[f"heads.{h[4:]}.{w}.weight"] = t
        elif k == "mlp_in.w":
            sd["mlp_in.weight"] = t
        elif k == "mlp_in.b":
            sd["mlp_in.bias"] = t
        elif k == "mlp_out.w":
            sd["mlp_out.weight"] = t
        elif k == "mlp_out.b":
            sd["mlp_out.bias"] = t
        else:
            raise KeyError(k)
    m.load_state_dict(sd)
    m.eval()
    return m


def main() -> None:
    torch.set_grad_enabled(False)
    print("== 输出端（logits）实际用到的频率：对 logits(a=0, b) 沿 b 做 DFT ==")
    print("   （c 取 softmax 概率最大的那一类；谱峰 = 网络真正在用的 k）\n")
    tops = {}
    for s in range(5):
        hdr, mats = load_bin(RUN / f"seed{s}" / "e025000.bin")
        model = build(mats, hdr["config"])
        a = torch.zeros(P, dtype=torch.long)
        b = torch.arange(P, dtype=torch.long)
        logits = model(a, b)                                     # [P, C]：每行一个 b
        y = logits[0].numpy()                                    # b=0 时各类的 logit
        spec = np.fft.fft(y)
        n = 1 + P // 2
        energy = np.abs(spec[:n]) ** 2
        order = np.argsort(-energy)[:6]
        share = 100 * energy[order].sum() / energy.sum()
        tops[s] = sorted(int(k) for k in order)
        pred = logits.argmax(dim=-1).numpy()
        acc = float((pred == (b.numpy() % P)).mean())
        print(f"seed {s}: 准确率(a=0) {acc:.3f}；logit_0(b) 的 top-6 频率 {tops[s]}，"
              f"能量占比 {share:.1f}%")

    print()
    base = set(tops[0])
    for s in range(1, 5):
        print(f"seed 0 与 seed {s} 的 top-6 频率重合 {len(base & set(tops[s]))}/6")
    union = set().union(*(set(v) for v in tops.values()))
    inter = set.intersection(*(set(v) for v in tops.values()))
    print(f"5 个 seed 合起来用到 {sorted(union)}")
    print(f"5 个 seed 共有的频率 {sorted(inter)}")


if __name__ == "__main__":
    main()
