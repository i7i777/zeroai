"""一次性工具：为 web/public/viz-data 下每个 (run, seed) 生成真实的测试集样本。

用法：python _gen_testpairs.py
说明：
- 使用与训练完全相同的 numpy 划分（make_modular_addition_data(p, frac, seed)）。
- 每个 seed 写 test_pairs.json：{kind,p,frac_train,seed,n_test_total,a:[..],b:[..],c:[..]}
  从中随机抽 N_SAMPLE 个测试对（带确定种子），供前端“测试”按钮取数。
- 不重训，只算数据划分，几秒完成。写入不干扰正在后台导出的快照。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../02.NeelNanda
sys.path.insert(0, str(ROOT / "python"))

from grokking.data import make_modular_addition_data  # noqa: E402

VIZ = ROOT / "web" / "public" / "viz-data"
N_SAMPLE = 800

rng_sel = __import__("numpy").random.default_rng(20260907)


def main() -> None:
    written = 0
    for run_dir in sorted(p for p in VIZ.iterdir() if p.is_dir()):
        for seed_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            idxp = seed_dir / "index.json"
            if not idxp.exists():
                continue
            try:
                idx = json.loads(idxp.read_text(encoding="utf-8"))
                cfg = idx.get("config", {})
                p = int(cfg["p"])
                frac = float(cfg.get("frac_train", 0.3))
                seed = int(idx.get("seed", seed_dir.name.replace("seed", "")))
            except Exception as exc:  # pragma: no cover
                print(f"skip {seed_dir}: {exc}", flush=True)
                continue
            data = make_modular_addition_data(p, frac, seed)
            te = data.test_mask.nonzero()[0]
            n = te.size
            take = min(N_SAMPLE, n)
            # 每个 (run,seed) 用确定的随机下标，保证文件稳定
            sub = rng_sel.choice(te, size=take, replace=False)
            sub.sort()
            a = [int(data.a[i]) for i in sub]
            b = [int(data.b[i]) for i in sub]
            c = [int(data.c[i]) for i in sub]
            out = {
                "kind": "numpy split (与训练互斥)",
                "p": p, "frac_train": frac, "seed": seed,
                "n_test_total": int(n), "n": take,
                "a": a, "b": b, "c": c,
            }
            (seed_dir / "test_pairs.json").write_text(
                json.dumps(out, ensure_ascii=False), encoding="utf-8")
            print(f"wrote {seed_dir}  n_test={n}  sample={take}", flush=True)
            written += 1
    print(f"done, {written} files", flush=True)


if __name__ == "__main__":
    main()
