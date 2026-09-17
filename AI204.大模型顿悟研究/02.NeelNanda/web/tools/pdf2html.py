#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 docs/ 下的论文 PDF 转成 HTML（默认 web/public/paper-en-pdf.html）。

【现在的定位：备用脚本】
站点里那个「本地完整版」页面（public/paper-en-local.html）已改由
`tools/localize_arxiv_html.py` 镜像 arXiv/ar5iv 的 LaTeXML 渲染结果生成 —— 公式、图表是真排版，
效果远比本脚本好。本脚本只作为**离线兜底**保留：万一哪天在线渲染源消失，
还能从本地 PDF 重新生成一份“能读、可搜索”的版本。它不会覆盖 paper-en-local.html。

只做「格式转换」：从本地已有的 PDF 里抽取文字与插图，重排成站点风格的 HTML。
不做人工校对，公式是「尽力而为」的纯文本近似（上下标会用 <sub>/<sup> 标出）。

用法（在仓库根目录或任意位置）：
    python "AI204.大模型顿悟研究/02.NeelNanda/web/tools/pdf2html.py"
可选参数：
    --pdf PATH   指定 PDF（默认自动找 ../docs/*.pdf）
    --out PATH   指定输出 HTML（默认 ../public/paper-en-pdf.html）
    --zoom 3     插图裁剪倍率（3 ≈ 216 dpi）

依赖：PyMuPDF（import pymupdf）
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import re
import sys
import unicodedata
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    import fitz as pymupdf  # 旧包名

HERE = Path(__file__).resolve().parent          # web/tools
WEB = HERE.parent                                # web
ROOT = WEB.parent                                # 02.NeelNanda
DOCS = ROOT / "docs"
PUBLIC = WEB / "public"
IMGDIR_NAME = "images/en"

# 正文/公式字体族（其余字体若占多数，判定为「图内文字」丢弃）
BODY_FAMS = ("NimbusRomNo9L", "CMR", "CMMI", "CMSY", "CMEX", "MSBM", "NimbusMonL")
MATH_FAMS = ("CMR", "CMMI", "CMSY", "CMEX", "MSBM")
DROP_TEXT = ("Published as a conference paper at ICLR 2023",
             "Under review as a conference paper at ICLR 2023")

RE_HEAD2 = re.compile(r"^([0-9]+|[A-Z])\s+([A-Z][A-Z0-9 ,\-:’'()/&\.]{6,})$")   # 12 pt 章标题
RE_HEAD3 = re.compile(r"^([0-9]+\.[0-9]+|[A-Z](\.[0-9]+)+|[0-9]+\.[0-9]+\.[0-9]+)\s+([A-Z][A-Z0-9 ,\-:’'()/&\.]{3,})$")
RE_CAPTION = re.compile(r"^(Figure|Table)\s+([0-9]+|[A-Z][0-9]*)\s*[:.]")
RE_PAGENUM = re.compile(r"^\d{1,3}$")
# 这些词 + 连字符通常是「复合词」而不是行尾断词（reverse-engineering / one-layer / state-of-the-art …）
KEEP_HYPHEN = {
    "reverse", "one", "two", "three", "four", "five", "well", "self", "semi", "pre", "post",
    "non", "multi", "sub", "super", "high", "low", "end", "state", "of", "up", "down", "left",
    "right", "off", "on", "co", "anti", "inter", "over", "under", "mid", "best", "no", "in", "out",
}


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def span_font(s: dict) -> str:
    return re.sub(r"\d+$", "", s.get("font", ""))


def block_chars(block: dict) -> int:
    return sum(_n_chars(s) for l in block.get("lines", []) for s in l["spans"])


def body_ratio(block: dict) -> float:
    """块内「论文正文字体」字符占比。"""
    tot = ok = 0
    for l in block.get("lines", []):
        for s in l["spans"]:
            n = _n_chars(s)
            tot += n
            if span_font(s).startswith(BODY_FAMS):
                ok += n
    return 1.0 if tot == 0 else ok / tot


def math_ratio(block: dict) -> float:
    tot = ok = 0
    for l in block.get("lines", []):
        for s in l["spans"]:
            n = _n_chars(s)
            tot += n
            if span_font(s).startswith(MATH_FAMS):
                ok += n
    return 0.0 if tot == 0 else ok / tot


def cluster_boxes(boxes, pad: float):
    """把相互靠近的矩形并成连通块（并查集）。boxes: list[pymupdf.Rect]"""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    grown = [pymupdf.Rect(b.x0 - pad, b.y0 - pad, b.x1 + pad, b.y1 + pad) for b in boxes]
    for i in range(n):
        for j in range(i + 1, n):
            if grown[i].intersects(grown[j]):
                union(i, j)
    groups: dict[int, pymupdf.Rect] = {}
    for i, b in enumerate(boxes):
        r = groups.get(find(i))
        groups[find(i)] = pymupdf.Rect(b) if r is None else r | b
    return [r for r in groups.values() if r.width > 1 and r.height > 1]


def esc(t: str) -> str:
    return html.escape(t, quote=False)


def clean_text(t: str, keep_edges: bool = False) -> str:
    """统一空白、把 PDF 里常见的私用区/异体字符换成普通字符。

    keep_edges=True 时保留首尾空格（逐 span 拼行时必须保留，否则词与词会粘在一起）。
    """
    t = t.replace("\u00ad", "")                     # soft hyphen
    t = t.replace("\ufb00", "ff").replace("\ufb01", "fi").replace("\ufb02", "fl")
    t = t.replace("\ufb03", "ffi").replace("\ufb04", "ffl")
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[\s\u00a0]+", " ", t)
    return t if keep_edges else t.strip()


# --------------------------------------------------------------------------- #
# 行 / 块 → HTML
# --------------------------------------------------------------------------- #
TAGMAP = {
    "m": ('<span class="m">', "</span>"),
    "strong": ("<strong>", "</strong>"),
    "em": ("<em>", "</em>"),
    "sub": ("<sub>", "</sub>"),
    "sup": ("<sup>", "</sup>"),
}


def _n_chars(span: dict) -> int:
    """span 里「可见字符」个数（rawdict 有 chars；dict 只有 text）。"""
    if span.get("chars"):
        return sum(1 for ch in span["chars"] if ch["c"].strip())
    return len(span.get("text", "").strip())


def line_has_text(line: dict) -> bool:
    return any(_n_chars(s) for s in line.get("spans", []))


def _line_baseline(line: dict) -> float:
    """行内「正常字号文字」的基线：按字符数投票，避开上下标的偏移。"""
    votes: dict[float, int] = {}
    for s in line["spans"]:
        y = round(s["origin"][1], 1)
        votes[y] = votes.get(y, 0) + max(1, _n_chars(s))
    return max(votes.items(), key=lambda kv: kv[1])[0]


def _style_of(span: dict, base_size: float, base_y: float) -> tuple:
    fam = span_font(span)
    size = span["size"]
    y = span["origin"][1]
    st = []
    if fam.startswith(MATH_FAMS):
        st.append("m")
    if size < base_size * 0.82 and abs(y - base_y) > 0.6:
        st.append("sup" if y < base_y else "sub")
    if "Medi" in fam or "Bold" in fam:
        st.append("strong")
    elif "Ital" in fam:
        st.append("em")
    return tuple(st)


def line_html(line: dict) -> str:
    """一行 → HTML。逐字符拼接，按字符间距补空格
    （PyMuPDF 的 span.text 在某些字体下会丢空格，所以不能直接用）。"""
    spans = [s for s in line["spans"] if _n_chars(s)]
    if not spans:
        return ""
    base_size = max(s["size"] for s in spans)
    base_y = _line_baseline(line)
    runs: list[tuple[tuple, str]] = []

    def push(style: tuple, txt: str) -> None:
        if runs and runs[-1][0] == style:
            runs[-1] = (style, runs[-1][1] + txt)
        else:
            runs.append((style, txt))

    prev_x1 = prev_size = None
    for s in spans:
        style = _style_of(s, base_size, base_y)
        chars = s.get("chars")
        if not chars:
            push(style, clean_text(s.get("text", ""), keep_edges=True))
            continue
        for ch in chars:
            c = ch["c"]
            if c in ("\n", "\r"):
                continue
            if prev_x1 is not None and c != " ":
                gap = ch["bbox"][0] - prev_x1
                if gap > max(0.2 * max(s["size"], prev_size or s["size"]), 0.9):
                    push((), " ")
            push(style, c)
            prev_x1 = ch["bbox"][2]
            prev_size = s["size"]

    out: list[str] = []
    for style, txt in runs:
        h = esc(clean_text(txt, keep_edges=True))
        if not h:
            continue
        for tag in ("m", "strong", "em", "sub", "sup"):
            if tag in style:
                o, c = TAGMAP[tag]
                h = o + h + c
        out.append(h)
    return "".join(out)


def block_lines(block: dict) -> list[dict]:
    return [l for l in block.get("lines", []) if line_has_text(l)]


def line_plain(line: dict) -> str:
    return re.sub(r"<[^>]+>", "", line_html(line))


def block_plain(block: dict) -> str:
    """块的纯文本（走同一个「按间距补空格」的行重建，供分类/匹配用）。"""
    return clean_text(" ".join(line_plain(l) for l in block_lines(block)))


def join_paragraph(lines: list[dict]) -> str:
    """把多行拼成一段。行尾连字符按「断词」处理：去掉连字符且不留空格。"""
    out = ""
    prev_plain = ""
    for l in lines:
        h = line_html(l).strip()
        if not h:
            continue
        plain = re.sub(r"<[^>]+>", "", h)
        if not out:
            out = h
        elif prev_plain.endswith(("-", "\u2010", "\u00ad")) and plain[:1].islower():
            # 保留末尾的收尾标签（</em> 之类），只动连字符本身
            tail_tag = re.search(r"((?:</(?:em|strong|span|sub|sup)>)*)$", out).group(1)
            core = out[: len(out) - len(tail_tag)].rstrip()
            word = re.split(r"[^A-Za-z]", core[:-1])[-1].lower()
            keep_dash = word in KEEP_HYPHEN          # 复合词：留连字符、只去掉换行带来的空格
            out = core[:-1] + ("-" if keep_dash else "") + tail_tag + h
        else:
            out += " " + h
        prev_plain = plain
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def convert(pdf_path: Path, out_path: Path, zoom: float) -> dict:
    doc = pymupdf.open(pdf_path)
    imgdir = PUBLIC / IMGDIR_NAME
    if imgdir.exists():
        for old in imgdir.glob("fig-*.png"):
            old.unlink()
    imgdir.mkdir(parents=True, exist_ok=True)

    body: list[str] = []            # HTML 片段（按页顺序）
    toc: list[tuple[int, str, str]] = []   # (level, id, 标题)
    stats = {"pages": doc.page_count, "figs": 0, "paras": 0, "eqs": 0,
             "headings": 0, "captions": 0, "dropped": 0}
    hid = 0

    for pno in range(doc.page_count):
        page = doc[pno]
        pw, ph = page.rect.width, page.rect.height
        blocks = page.get_text("rawdict")["blocks"]

        # ---- 1) 找插图/表格区域：矢量绘图 + 位图块 + 小字号文字块（表格正文）聚类 ----
        boxes = []
        for d in page.get_drawings():
            r = d["rect"]
            if r.width * r.height > 1.5:
                boxes.append(r)
        for b in blocks:
            if b.get("type") == 1:
                boxes.append(pymupdf.Rect(b["bbox"]))
                continue
            small = [s for l in b.get("lines", []) for s in l["spans"] if _n_chars(s)]
            if small and max(s["size"] for s in small) <= 9.4:      # 表格正文字号（题注是 10 pt）
                boxes.append(pymupdf.Rect(b["bbox"]))
        figs = []
        for c in cluster_boxes(boxes, pad=6):
            if c.width > 120 and c.height > 70 and c.width * c.height > 0.012 * pw * ph:
                figs.append(c)
        # 去掉被更大区域包含的框
        figs = [c for c in figs if not any(c != o and (o & c).get_area() > 0.98 * c.get_area() for o in figs)]
        figs.sort(key=lambda r: (round(r.y0, 1), r.x0))

        # ---- 2) 文本块过滤 ----
        keep: list[dict] = []
        for b in blocks:
            if b.get("type") != 0:
                continue
            lines = block_lines(b)
            if not lines:
                continue
            txt = block_plain(b)
            if not txt:
                continue
            if any(txt.startswith(d) for d in DROP_TEXT):           # 页眉
                stats["dropped"] += 1
                continue
            if RE_PAGENUM.match(txt) and (b["bbox"][1] < 70 or b["bbox"][3] > ph - 60):
                stats["dropped"] += 1                               # 页码
                continue
            if body_ratio(b) < 0.55:                                # 图内文字 / arXiv 水印
                stats["dropped"] += 1
                continue
            cx = (b["bbox"][0] + b["bbox"][2]) / 2
            cy = (b["bbox"][1] + b["bbox"][3]) / 2
            if any(r.x0 - 2 <= cx <= r.x1 + 2 and r.y0 - 2 <= cy <= r.y1 + 2 for r in figs):
                stats["dropped"] += 1                               # 落在插图区域内的文字
                continue
            keep.append(b)

        # ---- 3) 插图裁剪 → PNG，同时收集页内元素（kind, y0, y1, 内容）----
        items: list[list] = []
        for i, r in enumerate(figs):
            clip = pymupdf.Rect(max(r.x0, 0), max(r.y0, 0), min(r.x1, pw), min(r.y1, ph))
            pix = page.get_pixmap(clip=clip, matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            name = f"fig-p{pno + 1:02d}-{i + 1}.png"
            pix.save(imgdir / name)
            stats["figs"] += 1
            items.append(["fig", r.y0, r.y1,
                          f'<figure class="fig"><img src="/{IMGDIR_NAME}/{name}" '
                          f'alt="Figure (p.{pno + 1})" loading="lazy" /></figure>'])

        # ---- 4) 文本块分类 ----
        for b in keep:
            lines = block_lines(b)
            txt = block_plain(b)
            flat = re.sub(r"\s+", " ", txt)
            max_size = max(s["size"] for l in lines for s in l["spans"])
            n_lines = len(lines)
            y0, y1 = b["bbox"][1], b["bbox"][3]

            if max_size >= 15.0:                                     # 论文大标题（页面上方 h1 已给）
                stats["dropped"] += 1
                continue
            if max_size >= 11.4 and len(flat) <= 130:               # 章标题（12 pt）
                hid += 1
                anchor = f"en-s{hid}"
                title = re.sub(r"^([0-9A-Z]+)\s+", r"\1 ", flat)
                toc.append((2, anchor, title))
                stats["headings"] += 1
                items.append(["h2", y0, y1, f'<h2 id="{anchor}">{esc(title)}</h2>'])
                continue
            if n_lines <= 2 and RE_HEAD3.match(flat.replace("  ", " ")):   # 小节标题（10 pt 全大写）
                hid += 1
                anchor = f"en-s{hid}"
                toc.append((3, anchor, flat))
                stats["headings"] += 1
                items.append(["h3", y0, y1, f'<h3 id="{anchor}">{esc(flat)}</h3>'])
                continue
            if RE_CAPTION.match(flat) and n_lines <= 6:              # 图表题注
                stats["captions"] += 1
                items.append(["cap", y0, y1, lines])
                continue
            if math_ratio(b) > 0.5 and max_size >= 9.0 and len(flat) <= 200:   # 独立公式
                stats["eqs"] += 1
                items.append(["eq", y0, y1, lines])
                continue
            stats["paras"] += 1
            items.append(["p", y0, y1, lines])

        items.sort(key=lambda it: (round(it[1], 1), 0 if it[0] != "fig" else -1))

        # ---- 5) 合并被 PDF 拆散的同一段（行距级别的块边界 / 行尾连字符）----
        flow: list[list] = []
        for it in items:
            if flow and it[0] == "p" and flow[-1][0] == "p":
                prev = flow[-1]
                gap = it[1] - prev[2]
                prev_txt = re.sub(r"<[^>]+>", "", join_paragraph(prev[3])).rstrip()
                cont = prev_txt.endswith("-") or (gap < 13.0 and not re.search(r"[.!?:;]$", prev_txt))
                if gap < 13.5 and cont:
                    prev[3] = prev[3] + it[3]          # 行列表直接接上，交给 join_paragraph 处理连字符
                    prev[2] = max(prev[2], it[2])
                    continue
            flow.append(it)

        # ---- 6) 渲染 + 插图与其后的题注合成 <figure> ----
        merged: list[str] = []
        pending_fig = None
        for kind, _y0, _y1, payload in flow:
            if kind == "fig":
                if pending_fig:
                    merged.append(pending_fig)
                pending_fig = payload
                continue
            if kind == "cap" and pending_fig:
                merged.append(pending_fig[:-len("</figure>")] + f"<figcaption>{join_paragraph(payload)}</figcaption></figure>")
                pending_fig = None
                continue
            if pending_fig:
                merged.append(pending_fig)
                pending_fig = None
            if kind == "p":
                merged.append(f"<p>{join_paragraph(payload)}</p>")
            elif kind == "cap":
                merged.append(f"<figcaption>{join_paragraph(payload)}</figcaption>")
            elif kind == "eq":
                merged.append(f'<div class="eq">{join_paragraph(payload)}</div>')
            else:
                merged.append(payload)
        if pending_fig:
            merged.append(pending_fig)
        body.append(f'<div class="pdf-page" data-page="{pno + 1}">' + "".join(merged) + "</div>")

    # ---- 7) 组装整页 HTML ----
    joined = "\n".join(body)
    today = _dt.date.today().isoformat()
    toc_html = "\n".join(
        f'<li class="lvl{lvl}"><a href="#{a}">{esc(t)}</a></li>' for lvl, a, t in toc)
    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>论文英文原版（HTML） · Progress measures for grokking via mechanistic interpretability</title>
<link rel="stylesheet" href="/css/style.css" />
<style>
  .en-body {{ font-family: var(--font-serif); font-size: 16px; line-height: 1.72; }}
  .en-body p {{ margin: 0 0 13px; text-align: justify; hyphens: auto; }}
  .en-body h2 {{ font-size: 20px; letter-spacing: .01em; margin: 34px 0 12px; }}
  .en-body h3 {{ font-size: 16.5px; margin: 24px 0 10px; color: var(--accent); }}
  .en-body .m {{ font-family: var(--font-serif); font-style: italic; }}
  .en-body .eq {{ margin: 14px 0; padding: 10px 14px; text-align: center; background: var(--code-bg);
                  border-left: 3px solid var(--line); border-radius: 6px; font-size: 15px; overflow-x: auto; }}
  .en-body figure.fig {{ margin: 18px 0; text-align: center; }}
  .en-body figure.fig img {{ max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }}
  .en-body figure.fig figcaption {{ font-size: 13.5px; color: var(--ink-soft); margin: 8px 0 0; text-align: left; }}
  .en-body figcaption {{ font-size: 13.5px; color: var(--ink-soft); margin: 8px 0 18px; text-align: left; }}
  .pdf-page {{ border-top: 1px dashed var(--line); margin-top: 26px; padding-top: 4px; position: relative; }}
  .pdf-page::after {{ content: "PDF p." attr(data-page); position: absolute; right: 0; top: -22px;
                      font-size: 11px; color: #b9b4ab; font-family: var(--font-mono); }}
  .gen-note {{ font-size: 13px; color: var(--ink-soft); background: var(--accent-soft);
               border-radius: var(--radius); padding: 12px 14px; margin: 0 0 22px; }}
  .gen-note b {{ color: var(--accent); }}
  .en-src {{ margin-top: 40px; border-top: 1px solid var(--line); padding-top: 18px; font-size: 14px; }}
  .en-src h2 {{ font-size: 19px; }}
  .en-src ul {{ margin: 8px 0 14px; padding-left: 20px; }}
  .en-src li {{ margin: 5px 0; }}
  .en-src pre.bibtex {{ background: var(--code-bg); border: 1px solid var(--line); border-radius: var(--radius);
                        padding: 12px 14px; overflow-x: auto; font-family: var(--font-mono); font-size: 12.5px; }}
  .en-src .license {{ font-size: 13px; color: var(--ink-soft); border-left: 3px solid var(--line); padding-left: 12px; }}
</style>
</head>
<body>
<div id="progress"></div>

<header class="topbar">
  <div class="topbar-inner">
    <a class="brand" href="/">
      <span class="logo">悟</span>
      <span>顿悟现象研究 <small>· 论文英文原版（HTML）</small></span>
    </a>
    <nav class="main">
      <a href="/">首页</a>
      <a class="btn" href="/model.html">🧬 模型可视化</a>    </nav>
  </div>
</header>

<div class="wrap paper-layout">
  <aside class="paper-side">
    <span class="back"><a href="/">← 返回首页</a></span>
    <div class="toc-title">Contents</div>
    <nav id="paper-toc"><ol>
{toc_html}
    </ol></nav>
    <div style="padding:14px 10px 0; font-size:12.5px; color:var(--ink-soft); line-height:1.6;">
      原文：arXiv:2301.05217v3（ICLR 2023）<br />
      <a href="/paper.pdf" target="_blank">本地 PDF 副本</a><br />
      <a href="https://arxiv.org/html/2301.05217v3" target="_blank" rel="noopener">arXiv 官方 HTML 版</a>
    </div>
  </aside>

  <article class="paper en-body">
    <h1 class="title-zh" id="top">Progress measures for grokking via mechanistic interpretability</h1>
    <div class="authors">Neel Nanda · Lawrence Chan · Tom Lieberum · Jess Smith · Jacob Steinhardt</div>
    <div class="meta-line">
      ICLR 2023 (Spotlight) · arXiv:2301.05217v3 · 本页为<b>本地转换版</b>（离线兜底，由
      <code>web/tools/pdf2html.py</code> 从 PDF 自动生成，{today}）·
      <a href="https://arxiv.org/html/2301.05217v3" target="_blank" rel="noopener">arXiv 官方 HTML 版</a>
    </div>

    <p class="gen-note">
      <b>这是自动转换的 HTML 版</b>（离线可读）：文字与插图从本地 PDF 抽取后重排，插图按原文位置裁剪成图片；
      公式为纯文本近似（上下标用 <code>&lt;sub&gt;/&lt;sup&gt;</code> 表示，复杂公式可能与原排版有差异），
      编辑排版细节（分栏、浮动位置、脚注编号等）不保证与 PDF 完全一致。
      <br />
      <strong>想要排版正确的 HTML？→
      <a href="https://arxiv.org/html/2301.05217v3" target="_blank" rel="noopener">arXiv 官方 HTML 版</a>
      （公式/图表均为正常排版，推荐）。</strong>
      原文版权归作者所有（arXiv 非独占许可），本页仅供本地研读。
    </p>

{joined}

    <section class="en-src">
      <h2 style="border:none; padding:0;">Sources &amp; citation</h2>
      <ul>
        <li><strong>arXiv HTML (official, experimental)</strong> —
          <a href="https://arxiv.org/html/2301.05217v3" target="_blank" rel="noopener">arxiv.org/html/2301.05217v3</a></li>
        <li><strong>arXiv abstract page</strong> —
          <a href="https://arxiv.org/abs/2301.05217" target="_blank" rel="noopener">arxiv.org/abs/2301.05217</a></li>
        <li><strong>LaTeX source</strong> —
          <a href="https://arxiv.org/src/2301.05217" target="_blank" rel="noopener">arxiv.org/src/2301.05217</a></li>
        <li><strong>Local copy</strong> —
          <a href="/paper.pdf" target="_blank">/paper.pdf</a>（后端从 <code>docs/</code> 托管）</li>
        <li><strong>模型可视化</strong> — <a href="/model.html">/model.html</a></li>
      </ul>
      <pre class="bibtex">@inproceedings{{nanda2023progress,
  title     = {{Progress measures for grokking via mechanistic interpretability}},
  author    = {{Nanda, Neel and Chan, Lawrence and Lieberum, Tom and Smith, Jess and Steinhardt, Jacob}},
  booktitle = {{The Eleventh International Conference on Learning Representations (ICLR)}},
  year      = {{2023}},
  url       = {{https://arxiv.org/abs/2301.05217}}
}}</pre>
      <p class="license">
        Copyright of the original text remains with the authors (arXiv non-exclusive license).
        This page is a local format conversion of the PDF provided in <code>docs/</code> for study purposes;
        the text has <em>not</em> been copy-edited — for authoritative content use the
        <a href="/paper.pdf" target="_blank">PDF</a> or the
        <a href="https://arxiv.org/html/2301.05217v3" target="_blank" rel="noopener">official arXiv HTML</a>.
      </p>
    </section>
  </article>
</div>

<footer class="site">
  <div class="wrap" style="padding-bottom: 0;">
    <div>由 <code>web/tools/pdf2html.py</code> 从 <code>docs/</code> 的 PDF 自动生成 · 仅供学习 · 版权归原作者</div>
    <div class="api-line" id="api-line">正在连接后端 /api/meta …</div>
  </div>
</footer>

<button id="toTop" aria-label="回到顶部">↑</button>
<script src="/js/main.js"></script>
</body>
</html>
"""
    out_path.write_text(page_html, encoding="utf-8")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="论文 PDF → HTML")
    ap.add_argument("--pdf", default=None, help="输入 PDF（默认 docs/*.pdf）")
    ap.add_argument("--out", default=None, help="输出 HTML（默认 public/paper-en-pdf.html）")
    ap.add_argument("--zoom", type=float, default=3.0, help="插图裁剪倍率（默认 3 ≈ 216dpi）")
    args = ap.parse_args(argv)

    pdf = Path(args.pdf) if args.pdf else sorted(DOCS.glob("*.pdf"))[0]
    out = Path(args.out) if args.out else PUBLIC / "paper-en-pdf.html"
    if not pdf.exists():
        print(f"[pdf2html] 找不到 PDF：{pdf}", file=sys.stderr)
        return 1

    st = convert(pdf, out, args.zoom)
    print(f"[pdf2html] {pdf.name} → {out}")
    print(f"[pdf2html] 页 {st['pages']} · 章节标题 {st['headings']} · 段落 {st['paras']} · "
          f"公式块 {st['eqs']} · 插图 {st['figs']} · 题注 {st['captions']} · 丢弃(页眉/页码/图内文字) {st['dropped']}")
    print(f"[pdf2html] 插图目录：{PUBLIC / IMGDIR_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
