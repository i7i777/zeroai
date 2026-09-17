"""一次性探针（可删）：把结构图里画的那些“输出张量”在 PyTorch 参考实现里复算一遍，
用来核对 JS 侧 archTensors() 的峰值/形状是否一致。

运行：python _arch_tensor_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _seed_diff_probe2 import RUN, build, load_bin  # noqa: E402

A, B = 37, 70


def pk(name: str, arr: np.ndarray) -> None:
    flat = np.asarray(arr).reshape(-1)
    print("  %-30s shape=%-8s max|.|=%8.4f  nan=%d"
          % (name, str(np.asarray(arr).shape), float(np.abs(flat).max()), int(np.isnan(flat).sum())))


def main() -> None:
    torch.set_grad_enabled(False)
    hdr, mats = load_bin(RUN / "seed0" / "e025000.bin")
    model = build(mats, hdr["config"])
    a = torch.tensor([A])
    b = torch.tensor([B])
    logits, diag = model.forward_full(a, b)

    x = model.embed_inputs(a, b)
    head_outs = []
    for head in model.heads:
        h_out, _attn = head(x)                   # h_out: [B, T, d]
        head_outs.append(h_out[0].numpy())       # 每个头在 3 个位置上的输出

    print("seed0 e25000 · a=%d b=%d（下面数值应与页面上 MLP 块右侧的 max|·| 对上）" % (A, B))
    x0 = diag["x0"][0].numpy()
    for i in range(3):
        pk("EMBED 位置%d (x0)" % (i + 1), x0[i])
    for i, ho in enumerate(head_outs):
        pk("ATTENTION head%d 位置「=」" % i, ho[-1])
    attn_sum = np.sum([ho[-1] for ho in head_outs], axis=0)
    pk("ATTENTION Σ（位置「=」）", attn_sum)
    pk("MLP 输入 resid2", diag["resid2"][0].numpy())
    pk("MLP 隐层 ReLU", diag["mlp_acts"][0].numpy())
    out = diag["x2"][0].numpy() - diag["resid2"][0].numpy()
    pk("MLP 输出 W_out·h+b", out)
    pk("LOGITS", logits[0].numpy())
    lg = logits[0].numpy()
    print("  truth c=%d  pred=%d  logit[真值]=%.4f  max logit=%.4f"
          % ((A + B) % 113, int(lg.argmax()), lg[(A + B) % 113], lg.max()))


if __name__ == "__main__":
    main()
