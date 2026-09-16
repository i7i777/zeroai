"""一次性工具：从论文 PDF 里裁出 Figure 1（架构 + 傅里叶乘法示意图）。

用于 paper-zh.html 里"我们声称学到的网络使用如下算法（图 1）"这句话的配图
（原文 PDF 不含 Figure 1 的独立图片对象时，按"图注以上、正文以下的图形区域"裁剪）。

  python _extract_fig1.py --probe          # 只打印探测到的页面/区域，不写文件
  python _extract_fig1.py                  # 自动探测并导出
  python _extract_fig1.py --clip 60,120,540,430 --dpi 400   # 手动指定区域（单位 pt）

输出：web/public/images/fig1_architecture.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
NEEL = HERE.parent                       # 02.NeelNanda
PDF = NEEL / "docs" / "2301.05217_Progress_measures_for_grokking_via_mechanistic_interpretability.pdf"
OUT = NEEL / "web" / "public" / "images" / "fig1_architecture.png"

CAPTION = "Figure 1"
LOOKBACK = 760.0      # 只往上找这么多 pt 的图形内容
PAD = 4.0             # 裁剪外扩


def find_caption(page: "pymupdf.Page") -> "pymupdf.Rect | None":
    """返回图注 'Figure 1:' 的文字矩形。"""
    hits = page.search_for(CAPTION)
    for r in hits:
        # 只要图注那种"另起一行、位于左栏"的命中，排除正文中的交叉引用
        line = page.get_text("text", clip=pymupdf.Rect(r.x0 - 2, r.y0 - 1, page.rect.x1, r.y1 + 1))
        if line.strip().startswith(CAPTION):
            return r
    return hits[0] if hits else None


def content_bbox(page: "pymupdf.Page", caption: "pymupdf.Rect") -> "pymupdf.Rect | None":
    """图注正上方、最近一块连续图形内容（矢量绘图 + 位图）的包围盒。"""
    boxes: list[pymupdf.Rect] = []

    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.is_empty or r.height < 0.5 and r.width < 0.5:
            continue
        # 只要图注上方 LOOKBACK 以内的、且与该栏有横向重叠的绘图
        if r.y1 <= caption.y0 + 1 and r.y0 >= caption.y0 - LOOKBACK:
            boxes.append(r)

    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.y1 <= caption.y0 + 1 and r.y0 >= caption.y0 - LOOKBACK:
            boxes.append(r)

    if not boxes:
        return None

    # 从图注往上连续延伸：一旦出现大于 24pt 的纵向空隙就截断（避免吃进正文段落）
    boxes.sort(key=lambda r: r.y1, reverse=True)
    band = [boxes[0]]
    top = boxes[0].y0
    for r in boxes[1:]:
        if top - r.y1 > 24:
            break
        band.append(r)
        top = min(top, r.y0)
    return pymupdf.Rect(
        min(r.x0 for r in band), top,
        max(r.x1 for r in band), max(r.y1 for r in band),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(PDF))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--clip", help="手动区域 x0,y0,x1,y1（pt，页面左上为原点）")
    ap.add_argument("--dpi", type=float, default=300.0)
    ap.add_argument("--probe", action="store_true", help="只打印信息，不写文件")
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    print(f"PDF: {args.pdf}\n页数: {doc.page_count}\n")

    page = None
    caption = None
    for i, pg in enumerate(doc):
        if not pg.search_for(CAPTION):
            continue
        cap = find_caption(pg)
        print(f"[第 {i + 1} 页] 命中 {CAPTION}: caption={tuple(round(v, 1) for v in cap)}"
              f"  矢量图形={len(pg.get_drawings())}  位图={len(pg.get_image_info())}")
        if page is None and cap is not None:
            page, caption = pg, cap

    if page is None or caption is None:
        print("未找到 Figure 1，请改用 --clip 手动指定区域")
        return 1

    bbox = content_bbox(page, caption)
    print(f"\n自动探测区域: {tuple(round(v, 1) for v in bbox) if bbox else None}")
    if bbox is None:
        return 1

    clip = pymupdf.Rect(*(float(v) for v in args.clip.split(","))) if args.clip else bbox
    clip = pymupdf.Rect(clip.x0 - PAD, clip.y0 - PAD, clip.x1 + PAD, clip.y1 + PAD) & page.rect
    zoom = args.dpi / 72.0
    print(f"裁剪区域: ({clip.x0:.1f}, {clip.y0:.1f}) – ({clip.x1:.1f}, {clip.y1:.1f})"
          f"  = {clip.width:.0f}×{clip.height:.0f} pt @ {args.dpi:g} dpi"
          f"  → {clip.width * zoom:.0f}×{clip.height * zoom:.0f} px")

    if args.probe:
        return 0

    pix = page.get_pixmap(clip=clip, matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pix.save(out)
    print(f"已保存: {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
