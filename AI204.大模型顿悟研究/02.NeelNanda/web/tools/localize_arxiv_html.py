#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 arXiv / ar5iv 的 LaTeXML 渲染结果镜像成一份**本地完整版**页面。

为什么要这么做
--------------
`tools/pdf2html.py` 是从 PDF 里“抠字”重排的近似稿：公式不排、图表位置靠猜，读起来很别扭。
arXiv 官方 HTML 版与 ar5iv 都是 **LaTeXML** 从作者 LaTeX 源码编译出来的，公式（MathML）、
图表、章节、参考文献都是真正排好版的。本脚本把这种渲染结果的**正文 DOM + CSS + 图片**
下载到 `web/public/` 下，改写成站内路径，于是得到一个：

* 排版正确（和官方 HTML 版同源）；
* 完全离线（除 Google Fonts 走 CDN，脱网时自动回落到系统衬线字体）；
* 可全文搜索 / 复制；
* 顶上带本站导航与出处说明。

输出（默认两份内容相同）：

* `public/index.html` —— **站点首页就是这篇论文**
* `public/paper-en-local.html` —— 同样的内容，老链接/书签仍可用

用法
----
    python "AI204.大模型顿悟研究/02.NeelNanda/web/tools/localize_arxiv_html.py"

    --url   源页面（默认 ar5iv 渲染；也可换成 https://arxiv.org/html/2301.05217v3）
    --out   只写指定的一份输出

说明
----
* 正文版权归作者所有（arXiv 非独占许可）。本脚本只做**格式搬运**，不做任何改写；
  页面顶部与底部都标注了原始出处，仅供本地学习使用。
* 重复运行会覆盖输出文件与 `public/assets/paper-en/` 下的图片。
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
PUBLIC = WEB / "public"

DEFAULT_URL = "https://ar5iv.labs.arxiv.org/html/2301.05217"
ARXIV_HTML = "https://arxiv.org/html/2301.05217v3"
ARXIV_ABS = "https://arxiv.org/abs/2301.05217"

CSS_DIR = PUBLIC / "assets" / "ar5iv"        # LaTeXML 样式表 + logo
IMG_DIR = PUBLIC / "assets" / "paper-en"     # 论文插图（保留 figs/ 子目录）

UA = "Mozilla/5.0 (compatible; zeroai-local-mirror/1.0)"

SITEBAR = """<div class="sitebar">
  <div class="sitebar-inner">
    <a class="sb-brand" href="/"><span class="sb-logo">悟</span>顿悟现象研究 <small>· 论文英文原文（本地完整版）</small></a>
    <nav class="sb-nav">
      <a class="sb-btn ghost" href="/reverse.html">🔬 逆向工程</a>
      <a class="sb-btn" href="/model.html">🧬 模型可视化</a>
    </nav>
  </div>
</div>
<div class="sitebar-note">
  <strong>本地完整版（本站首页）</strong>：正文、公式（MathML）、图表、参考文献都由 LaTeXML 编译而成，是真正的排版
  ——不是从 PDF 抠字重排的近似稿；整页已镜像到本机，离线可用、可全文搜索与复制。<br />
  原始出处：<a href="{src}" target="_blank" rel="noopener">{src}</a>
  · 摘要页：<a href="{abs_}" target="_blank" rel="noopener">{abs_}</a>。
  英文正文版权归作者所有（arXiv 非独占许可），本页仅为本地学习镜像。
</div>
"""

FOOTER = """<footer class="sitebar-foot">
  <div class="sitebar-inner">
    <div>
      正文由 LaTeXML 生成（源：<a href="{src}" target="_blank" rel="noopener">ar5iv</a> /
      <a href="{official}" target="_blank" rel="noopener">arXiv 官方 HTML</a>），本页为其本地镜像 ·
      生成的 HTML 请勿手工修改（见 <code>web/tools/localize_arxiv_html.py</code>）
    </div>
    <div class="sb-foot-line">本地镜像生成于 {today} · 仅供学习 · 版权归原作者</div>
  </div>
  <button class="sb-totop" type="button" onclick="window.scrollTo({{top:0,behavior:'smooth'}})">↑ 回到顶部</button>
</footer>
"""

HEAD_STYLE = """<style>
  /* 本站外壳（不动正文样式，正文交给 ar5iv 自带的 CSS） */
  .sitebar { position: sticky; top: 0; z-index: 60; background: #fffefb; border-bottom: 1px solid #e4e2dd; }
  .sitebar-inner {
    max-width: 1180px; margin: 0 auto; padding: 10px 20px; display: flex; flex-wrap: wrap;
    gap: 12px; align-items: center; justify-content: space-between;
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .sb-brand { display: flex; align-items: center; gap: 9px; text-decoration: none; color: #1f2328; font-weight: 600; font-size: 15px; }
  .sb-brand small { color: #7a8290; font-weight: 400; font-size: 12.5px; }
  .sb-logo {
    display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
    border-radius: 7px; background: #8b2f2f; color: #fff; font-size: 14px;
  }
  .sb-nav { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 14px; }
  .sb-nav a { color: #35404d; text-decoration: none; border: 0; }
  .sb-nav a:hover { color: #8b2f2f; }
  .sb-nav a.sb-btn { background: #8b2f2f; color: #fff; padding: 5px 12px; border-radius: 999px; }
  .sb-nav a.sb-btn.ghost { background: #fff; color: #8b2f2f; border: 1px solid #e6cfcb; }
  .sitebar-note {
    max-width: 1180px; margin: 14px auto 6px; padding: 11px 14px; background: #fff;
    border: 1px solid #e4e2dd; border-left: 3px solid #8b2f2f; border-radius: 8px;
    font-size: 13.5px; line-height: 1.7; color: #57606a;
    font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", -apple-system, "Segoe UI", Roboto, sans-serif;
  }
  .sitebar-note a { color: #8b2f2f; }
  .sitebar-foot { border-top: 1px solid #e4e2dd; margin-top: 28px; padding: 18px 0 26px; background: #faf9f6; }
  .sitebar-foot .sitebar-inner { display: block; font-size: 13px; color: #6b7280; line-height: 1.8; }
  .sitebar-foot a { color: #8b2f2f; }
  .sb-foot-line { color: #9aa1ab; margin-top: 4px; }
  .sb-totop {
    display: block; margin: 0 auto 0; max-width: 1180px; padding: 6px 14px; cursor: pointer;
    border: 1px solid #e4e2dd; background: #fff; border-radius: 999px; color: #57606a; font-size: 13px;
  }
  .sb-totop:hover { border-color: #8b2f2f; color: #8b2f2f; }
</style>
"""


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def origin(url: str) -> str:
    m = re.match(r"(https?://[^/]+)", url)
    return m.group(1) if m else ""


def download(src_url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = fetch(src_url)
    dest.write_bytes(data)
    return len(data)


def build(url: str, outs: list[Path]) -> dict:
    html = fetch(url).decode("utf-8", "replace")
    base = origin(url)
    stats = {"css": 0, "img": 0, "bytes": 0, "fence": 0, "chrome": 0}

    # ---- 1. 正文 DOM：只取 <body> 内部，head 里的 og/canonical 等一律丢弃 ----
    body = html.split("<body>", 1)[1].rsplit("</body>", 1)[0]

    # 去掉 ar5iv 的亮/暗主题切换（本站外壳固定浅色）
    body = re.sub(r"<script>\s*function detectColorScheme\(\).*?</script>", "", body, flags=re.S)

    # 修 LaTeXML(0.7.5 oxide) 的老毛病：\left( \right) 被渲染成字面量 OPEN / CLOSE
    body, n_fence = re.subn(r">OPEN<", ">(<", body)
    body, n2 = re.subn(r">CLOSE<", ">)<", body)
    stats["fence"] = n_fence + n2

    # 清掉 ar5iv 自己的站点外壳：上一篇/下一篇/Feeling lucky/转换报告 等都是相对链接，
    # 镜像到本地后只会 404（本文正文不受影响）
    body, n_footer = re.subn(r'<div class="ar5iv-footer">.*?</div>', "", body, flags=re.S)
    # 亮/暗主题按钮的函数已被剔除，留着按钮会报错
    body, n_toggle = re.subn(r'<a class="ar5iv-toggle-color-scheme".*?</a>', "", body, flags=re.S)
    stats["chrome"] = n_footer + n_toggle

    # ---- 2. 样式表：下载到 public/assets/ar5iv/ 并改成站内绝对路径 ----
    css_hrefs = re.findall(r'<link[^>]*rel="stylesheet"[^>]*href="([^"]+)"', html)
    for href in css_hrefs:
        name = href.rsplit("/", 1)[-1]
        dest = CSS_DIR / name
        src = href if href.startswith("http") else base + href
        stats["bytes"] += download(src, dest)
        stats["css"] += 1
        html = html.replace(f'href="{href}"', f'href="/assets/ar5iv/{name}"')

    # ---- 3. 图片：/html/2301.05217/assets/** 与 /assets/ar5iv.png ----
    def rel_to_local(u: str) -> Path:
        if u.startswith("/html/"):
            return IMG_DIR / u.split("/assets/", 1)[1]          # 保留 figs/ 子目录
        return CSS_DIR / u.rsplit("/", 1)[-1]                    # 站点 logo 等

    assets = set(re.findall(r'(?:src|href)="(/html/2301\.05217/assets/[^"]+)"', body))
    assets |= set(re.findall(r'(?:src|href)="(/assets/[^"]+\.(?:png|jpg|svg|gif))"', body))
    for a in sorted(assets):
        dest = rel_to_local(a)
        stats["bytes"] += download(base + a, dest)
        stats["img"] += 1
        target = ("/assets/paper-en/" + a.split("/assets/", 1)[1]) if a.startswith("/html/") else f"/assets/ar5iv/{dest.name}"
        body = body.replace(f'"{a}"', f'"{target}"')

    # ---- 4. 拼出成品页面 ----
    page = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Progress measures for grokking via mechanistic interpretability · 英文原文（本地完整版）</title>
{"".join(f'<link media="all" rel="stylesheet" href="/assets/ar5iv/{h.rsplit("/", 1)[-1]}">' for h in css_hrefs)}
<link media="all" rel="stylesheet" href="/assets/paper-zh.css">
{HEAD_STYLE}</head>
<body>
{SITEBAR.format(src=url, official=ARXIV_HTML, abs_=ARXIV_ABS)}
{body}
{FOOTER.format(src=url, official=ARXIV_HTML, today=date.today().isoformat())}
<!-- 每段末尾的「T」按钮：点击在该段的英文原文 / 中文机翻之间切换（译文由 tools/translate_paper.py 生成） -->
<script src="/js/paper-zh.js" defer></script>
</body>
</html>
"""
    stats["outs"] = []
    for dest in outs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page, encoding="utf-8")
        stats["outs"].append(dest)
    stats["size"] = len(page.encode("utf-8"))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="镜像 arXiv/ar5iv 的 LaTeXML HTML 渲染为本地页面（默认写入站点首页）")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"源页面（默认 {DEFAULT_URL}）")
    ap.add_argument("--out", default=None,
                    help="只写这个输出路径（默认写 public/index.html 与 public/paper-en-local.html 两份）")
    args = ap.parse_args()

    outs = [Path(args.out)] if args.out else [PUBLIC / "index.html", PUBLIC / "paper-en-local.html"]
    try:
        st = build(args.url, outs)
    except Exception as exc:  # noqa: BLE001 —— 命令行工具，直接报错退出即可
        print(f"[localize] 失败：{exc}", file=sys.stderr)
        return 1

    for f in st["outs"]:
        print(f"[localize] → {f.relative_to(WEB)}")
    print(f"[localize] 源：{args.url}")
    print(f"[localize] 样式 {st['css']} 个 · 图片 {st['img']} 张 · 下载 {st['bytes'] / 1024:.0f} KB · "
          f"修正括号占位 {st['fence']} 处 · 清理站点外壳 {st['chrome']} 处 · 输出 {st['size'] / 1024:.0f} KB")
    print(f"[localize] 资源目录：{IMG_DIR.relative_to(WEB)}、{CSS_DIR.relative_to(WEB)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
