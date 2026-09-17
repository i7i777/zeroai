"""把论文页（`public/index.html`）里的每个段落 / 图注用本地 MT 模型译成中文，
产出 `public/assets/paper-zh.json`，供页面上每段的「T」小按钮切换显示。

用法（在 `web/` 目录下）：
    python tools/translate_paper.py --list            # 只列出提取到的块，不翻译
    python tools/translate_paper.py --limit 5         # 试译 5 块，看看效果
    python tools/translate_paper.py                   # 全部翻译（已有译文会跳过，可反复运行续译）
    python tools/translate_paper.py --model nllb      # 换 NLLB-200-distilled-600M（质量更好、更慢）

设计要点
--------
* **按 DOM 顺序**抽出块（`p.ltx_p` 与 `figcaption.ltx_caption`），JS 侧用同样的选择器顺序按下标取译文，
  因此不依赖 id 是否唯一（44 个图注没有独立 id）。
* 段落里的行内内容按“原子”保护：`<math>` / `<cite>` / `<a class="ltx_ref">` / `<img>` / `ltx_tag`
  原样保留；`<span class="ltx_font_bold">`、`<em>`、`<strong>` 保留标签但把里的文字一起翻译；
  其余包装标签（如 `ltx_text`）拆掉只留文字，免得把标记喂给翻译模型。
* 原子在待译文本里换成 `[[1]]` 这样的占位符；译文里若占位符丢失（模型吃掉了），该块**回退成英文原文**
  并在日志里报出来，绝不让公式被静静弄丢。
* 超长段落按句子切块（Marian 单次输入上限约 512 token），逐块翻译后按顺序拼回。
* 进度写在同一个 JSON 里（key = 块下标#分块下标），中断后重跑会跳过已完成的。

模型（默认 `Helsinki-NLP/opus-mt-en-zh`，约 300 MB）从 HuggingFace 镜像下载到本地缓存；
镜像地址由 `HF_ENDPOINT` 控制（默认 `https://hf-mirror.com`）。
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")     # 国内可达的镜像
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parent
WEB = HERE.parent
PUBLIC = WEB / "public"

MODELS = {
    "marian": "Helsinki-NLP/opus-mt-en-zh",
    "tcbig": "Helsinki-NLP/opus-mt-tc-big-en-zh",
    "nllb": "facebook/nllb-200-distilled-600M",
}

# 占位符：实测 Marian 会把 `[[1]]` 压成 `[1]`，所以直接用单中括号形式最稳


def mark(n: int) -> str:
    return f"[{n}]"

TAG_RE = re.compile(r"<(/?)([a-zA-Z][\w:-]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*?)(/?)>")
VOID_TAGS = {"br", "hr", "img", "svg"}
KEEP_TAGS = {"math", "cite", "table", "tr", "td", "th"} | VOID_TAGS
TRANS_TAGS = {"em", "strong", "b", "i"}
TRANS_CLASSES = ("ltx_font_bold", "ltx_font_italic")


def attrs_of(raw: str) -> dict[str, str]:
    return {k: v for k, v in re.findall(r'([\w:-]+)\s*=\s*"([^"]*)"', raw)}


def classify(tag: str, raw: str) -> str:
    """该标签怎么处理：'keep' 原样保留 | 'trans' 保留标签但翻译内部文字 | 'drop' 拆掉只留子内容。"""
    cls = attrs_of(raw).get("class", "")
    if tag == "a" and "ltx_ref" in cls:
        return "keep"          # 指向图/表/章节/文献的引用，原样留着
    if tag == "span" and "ltx_tag" in cls:
        return "keep"          # 「Figure 1:」这类标签
    if tag in KEEP_TAGS:
        return "keep"
    if tag in TRANS_TAGS or any(c in cls for c in TRANS_CLASSES):
        return "trans"
    return "drop"


def annotate(inner: str) -> tuple[str, list[tuple[str, str, str]]]:
    """块内容 → (带占位符的纯文本, 原子列表)。

    原子 = (处理方式, 原始 HTML, 原子内部纯文字)。占位符写成 `[[i]]`（i 从 1 开始）。
    """
    atoms: list[tuple[str, str, str]] = []
    out: list[str] = []
    collecting: dict | None = None      # 正在收集的 keep/trans 元素
    stack: list[str] = []               # 收集过程中尚未闭合的内层标签
    i = 0

    def add_atom(mode: str, raw: str, mark_it: bool = True) -> None:
        text = html_mod.unescape(re.sub(r"<[^>]+>", "", raw)).strip()
        atoms.append((mode, raw, text))
        if mark_it:
            out.append(mark(len(atoms)))

    while i < len(inner):
        m = TAG_RE.search(inner, i)
        if m is None:
            if collecting is None:
                out.append(re.sub(r"\s+", " ", inner[i:]))
            break
        closing, tag, raw, selfclose = m.group(1), m.group(2).lower(), m.group(3), m.group(4)

        if collecting is None:
            out.append(re.sub(r"\s+", " ", inner[i:m.start()]))
            if not closing:
                mode = classify(tag, raw)
                if mode == "drop":
                    i = m.end()
                    continue
                if tag in VOID_TAGS or selfclose:
                    add_atom(mode if mode == "keep" else "keep", m.group(0))
                    i = m.end()
                    continue
                collecting = {"mode": mode, "start": m.start(), "tag": tag}
                stack = []
        else:
            if not closing and not selfclose:
                stack.append(tag)
            elif closing:
                if stack and stack[-1] == tag:
                    stack.pop()
                elif not stack and tag == collecting["tag"]:
                    add_atom(collecting["mode"], inner[collecting["start"]:m.end()])
                    collecting = None
                elif tag in stack:                      # 容错：标签不匹配时弹到匹配位置
                    while stack and stack.pop() != tag:
                        pass
        i = m.end()

    if collecting is not None:                          # 没闭合（容错）
        add_atom(collecting["mode"], inner[collecting["start"]:])
    return "".join(out), atoms


def is_math(raw: str) -> bool:
    return raw.lstrip().lower().startswith("<math")


def restore(zh: str, atoms: list[tuple[str, str, str]], zh_atoms: dict[int, str]) -> tuple[str | None, list[int]]:
    """把占位符换回原 HTML。返回 (成品, 丢掉的公式序号)。

    公式丢了 → 返回 (None, [...])，该块回退成英文（不能让公式静静消失）；
    引用/图表编号丢了 → 把它们原样附到段尾（它们本来就多在句末），不拖累整段中文。
    """
    missing = [n for n in range(1, len(atoms) + 1) if mark(n) not in zh]
    lost_math = [n for n in missing if is_math(atoms[n - 1][1])]
    if lost_math:
        return None, lost_math

    tail: list[str] = []
    for n in missing:
        mode, raw, txt = atoms[n - 1]
        if mode == "trans" and zh_atoms.get(n):
            tail.append(zh_atoms[n])
        elif txt:
            tail.append(raw)

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if n < 1 or n > len(atoms):
            return m.group(0)
        mode, raw, _t = atoms[n - 1]
        if mode == "trans" and zh_atoms.get(n):
            gt = raw.index(">")
            return f"{raw[:gt + 1]}{zh_atoms[n]}{raw[raw.rindex('<'):]}"
        return raw

    zh = re.sub(r"\[(\d+)\]", repl, zh)
    if tail:
        zh = zh.rstrip() + " " + " ".join(tail)
    zh = re.sub(r"\s+([，。、；：）】》])", r"\1", zh)      # 顺手清掉中文字前的空格
    zh = re.sub(r"([（【《])\s+", r"\1", zh)
    return zh, []


SENT_RE = re.compile(r"(?<=[.!?])\s+")

# 机翻常见硬伤的词表修正（在拼装阶段做，改完只要重跑 --stage assemble，不用重译）
POLISH: list[tuple[str, str]] = [
    # —— 术语 ——（顺序有意义：先修“把 grokking 译成车”这类硬伤，再做泛化替换）
    (r"火车", "训练"),                     # train accuracy / train loss 被译成“火车…”
    (r"滑(?=的|机制|现象)", "顿悟"),        # grokking 的另一类误译
    (r"车", "顿悟"),                       # 本论文里单独出现的“车”只可能是 grokking 的误译
    (r"着[.。]", "顿悟。"),                  # 加粗小标题 “Grokking.” 被译成 “着.”
    (r"机械(?=解释)", "机制"),
    (r"扭转(?:网络的)?机制", "逆向工程网络的机制"),
    (r"(?:过于|太)合适", "过拟合"),
    (r"模块化添加", "模加法"),
    (r"通用(?=解决|算法)", "泛化"),
    (r"通用化", "泛化"),
    (r"过度适应|过适应", "过拟合"),
    (r"绩效", "性能"),
    (r"诱导偏见|归纳偏差", "归纳偏置"),
    (r"复杂性", "复杂度"),
    (r"减免(?=转化为|为|成)", "归约"),
    (r"限制(?=和|与|损失|模型|架构)", "受限"),
    (r"后(?=机制可解释性)", "事后"),
    (r"辩称", "认为"),
    (r"限制(?=损失|模型|架构)", "受限"),
    (r"(?:进步措施|进展措施|进步度量)", "进展度量"),
    (r"(?:比尼斯标准|弗罗本尼斯范数|Frobenius 标准)", "弗罗贝尼乌斯范数"),
    (r"(?:Fourier|傅立叶|傅里叶)(?:变化|变换|转換)", "傅里叶变换"),
    (r"网络重量", "网络权重"),
    (r"神经逻辑地图", "神经元–logits 映射"),
    (r"都删除", "都消融"),
    (r"删除所有关键频率", "消融所有关键频率"),
    (r"随机步行|随机行走|随机走路", "随机游走"),
    (r"体重(?:降低|衰减)", "权重衰减"),
    (r"模块(?=\s*(?:\[\d+\]|PP|P\b|p\b))", "mod "),   # mod P 的 P 常常是公式原子（占位符），也要认
    (r"从记忆到通用", "从记忆切换到泛化"),
    (r"直观的(?=，)", "直觉上"),
    (r"前 token 头", "前一个 token 头"),
    (r"感应头", "归纳头"),
    (r"(?<!批)规范化", "正则化"),
    (r"隐含", "隐式"),
    (r"阶段图表", "相图"),
    (r"复杂机械", "复杂结构"),
    (r"逻辑轴", "logits 轴"),
    (r"(?:训练|测试)精度", "准确率"),
    (r"质量(?=行为|变化|转变|差异|上)", "质上"),
    (r"棋牌", "国际象棋"),
    (r"开放理论", "开局理论"),
    (r"(减去|删去|消除|删除)(?=损失|关键频率|这些频率|所有频率)", "消融"),
    (r"其他(?:减去|消除)", "其他消融"),
    (r"未(?:减去|消融)(?=的)", "未做消融的"),
    (r"(?:新兴|出现)(?=行为)", "涌现"),
    (r"质量新能力|质新能力", "质上全新的能力"),
    (r"机械解释性|机械可解释性", "机制可解释性"),
    (r"(?:福利尔|傅立叶)", "傅里叶"),
    (r"(?:三角形识别|三角识别|三角特性|三角形特征)", "三角恒等式"),
    (r"代币", "token"),
    (r"(?:模块化加|模数加法|模块添加|模块化加法|模块化任务)", "模加法"),
    (r"减免(?=损失|后的)", "归约"),
    (r"转为循环|变成循环", "转换为旋转"),
    (r"空间重量", "权重"),
    (r"培训", "训练"),
    (r"(?:跨机体损失|交叉机能丧失|跨层缩损失|跨层缩|交叉熵丢失|交互熵损失|交义熵)", "交叉熵损失"),
    (r"(?:思想链|思路链|思维链)", "思维链"),
    (r"对数(?=值|中|的)", "logits"),
    (r"(?:单层变压器|一层变压器|单层转换器|单层变形器)", "单层 Transformer"),
    (r"变压器", "Transformer"),
    (r"(?:重现|再现)(?=行为|现象)", "涌现"),
    (r"嵌入式矩阵", "嵌入矩阵"),
    (r"([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", r"\1"),   # 中文之间多余的英文空格
    (r"概括", "泛化"),
    (r"(?<=[\u4e00-\u9fff])!", "！"),
    (r"(?<=[\u4e00-\u9fff])\?", "？"),
    (r"[ \t\u00a0]*（[ \t\u00a0]*", "（"),
    (r"[ \t\u00a0]*）", "）"),
    (r"）[ \t\u00a0]*(?=[\u4e00-\u9fff])", "）"),
]


def polish(zh: str) -> str:
    """把机翻结果收拾干净：先改词表里的硬伤，再统一标点与中英文之间的空格。"""
    for pat, rep in POLISH:
        zh = re.sub(pat, rep, zh)
    zh = re.sub(r"\s+([.,:;!?])", r"\1", zh)
    zh = re.sub(r"(?<=[\u4e00-\u9fff\]\)])\.(?=\s|$|[\u4e00-\u9fff])", "。", zh)
    zh = re.sub(r"(?<=[\u4e00-\u9fff\]\)]),\s*", "，", zh)
    zh = re.sub(r"(?<=[\u4e00-\u9fff\]\)]):\s*", "：", zh)
    zh = re.sub(r"(?<=[\u4e00-\u9fff\]\)]);\s*", "；", zh)
    zh = re.sub(r"\((?=[^)]*[\u4e00-\u9fff])", "（", zh)
    zh = re.sub(r"(?<=[\u4e00-\u9fff])\)", "）", zh)
    zh = re.sub(r"(?<=[A-Za-z0-9%])(?=[\u4e00-\u9fff])", " ", zh)      # Transformer是 → Transformer 是
    zh = re.sub(r"(?<=[\u4e00-\u9fff])(?=[A-Za-z0-9])", " ", zh)
    zh = re.sub(r"([\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", r"\1", zh)  # 中文之间多余的英文空格
    return zh.strip()


SEG_RE = re.compile(r"(\[\d+\])")


def split_segments(text: str) -> list[str]:
    """把带占位符的文本按占位符切成 ‹文字, [n], 文字, [n] …› 交替的片段（偶数位是文字）。

    整段翻译时占位符容易被模型吃掉；退到这种「按公式切开逐段翻」的模式后，
    占位符是自己拼回去的，因此公式必定不会丢。
    """
    return SEG_RE.split(text)


def chunk(text: str, limit: int = 320) -> list[str]:
    """按句子切块，块内大致不超过 limit 个词（Marian 单次输入上限 512 token）。"""
    parts = [p for p in SENT_RE.split(text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in parts:
        if cur and len(cur.split()) + len(p.split()) > limit:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [text]


# 短语词典：`<em>` / 加粗小标题这类 "trans" 原子太短，模型容易翻飞（实测 grokking → “的,”），
# 所以对常见短语直接给定译法（拼装阶段生效，英文原文大小写不敏感、忽略尾部的句号）
GLOSSARY: dict[str, str] = {
    "grokking": "顿悟",
    "grok": "顿悟",
    "progress measures": "进展度量",
    "progress measure": "进展度量",
    "mechanistic interpretability": "机制可解释性",
    "mechanistic explanations": "机制可解释性解释",
    "mechanistic explanation": "机制可解释性解释",
    "memorization": "记忆",
    "circuit formation": "电路形成",
    "cleanup": "清理",
    "generalization": "泛化",
    "generalizing solution": "泛化解",
    "restricted loss": "受限损失",
    "excluded loss": "排除损失",
    "phase changes": "阶段变化",
    "emergence": "涌现",
    "fourier space": "傅里叶空间",
    "trigonometric identities": "三角恒等式",
    # 章节标题
    "introduction": "引言",
    "related work": "相关工作",
    "background": "背景",
    "methods": "方法",
    "results": "实验结果",
    "discussion": "讨论",
    "conclusion": "结论",
    "conclusions": "结论",
    "limitations": "局限性",
    "appendix": "附录",
    "abstract": "摘要",
    "references": "参考文献",
    "acknowledgements": "致谢",
    "acknowledgments": "致谢",
}


def glossary(text: str) -> str | None:
    """查短语词典（忽略大小写与句尾标点）。"""
    key = text.strip().strip(".。").lower()
    return GLOSSARY.get(key)


class Translator:
    """本地 seq2seq 翻译（Marian / NLLB），CPU 上按批跑。"""

    def __init__(self, key: str = "marian", batch: int = 8) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer    # 延迟导入

        self.key, self.batch, self.name = key, batch, MODELS[key]
        print(f"载入模型 {self.name}（首次运行会从 {os.environ['HF_ENDPOINT']} 下载）…", flush=True)
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.name)
        self.model.eval()
        self.zh_id = None
        if key == "nllb":
            self.tok.src_lang = "eng_Latn"
            self.zh_id = self.tok.convert_tokens_to_ids("zho_Hans")

    def __call__(self, texts: Iterable[str]) -> list[str]:
        import torch

        texts = list(texts)
        out: list[str] = []
        for i in range(0, len(texts), self.batch):
            part = texts[i:i + self.batch]
            enc = self.tok(part, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=512, num_beams=1,
                                          forced_bos_token_id=self.zh_id)
            out.extend(t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True))
            print(f"  翻译中… {min(i + self.batch, len(texts))}/{len(texts)}", flush=True)
        return out


def iter_blocks(page: str) -> list[tuple[str, int, str]]:
    """按文档顺序抽出 (标签, 起始偏移, 内容 HTML)。

    覆盖：段落 `p.ltx_p`、图注 `figcaption.ltx_caption`、标题 `h1…h6.ltx_title`。
    顺序必须与 `public/js/paper-zh.js` 里的选择器顺序一致（都是文档顺序）。
    """
    found: list[tuple[str, int, str]] = []
    pats = [(r'<p id="[^"]*" class="ltx_p"[^>]*>', "p"),
            (r'<figcaption class="[^"]*ltx_caption[^"]*"[^>]*>', "figcaption")]
    pats += [(rf'<h{n} class="[^"]*ltx_title[^"]*"[^>]*>', f"h{n}") for n in range(1, 7)]
    for pat, tag in pats:
        for m in re.finditer(pat, page):
            end = page.find(f"</{tag}>", m.end())
            if end >= 0:
                found.append((tag, m.start(), page[m.end():end]))
    found.sort(key=lambda b: b[1])
    return found


def blk_key(inner: str) -> str:
    """块的稳定标识：内容哈希。

    用它做缓存 key（而不是下标）——这样以后往页面里插新块（比如加个标题）不会把
    已有 211 段的译文全打乱位置。
    """
    norm = re.sub(r"\s+", " ", inner).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=str(PUBLIC / "index.html"))
    ap.add_argument("--out", default=str(PUBLIC / "assets" / "paper-zh.json"))
    ap.add_argument("--cache", default=str(PUBLIC / "assets" / "paper-zh.cache.json"),
                    help="逐条原始机翻的缓存（保留下来，方便只重跑拼装阶段）")
    ap.add_argument("--overrides", default=str(HERE / "paper-zh-overrides.json"),
                    help="人工校订层：{块下标: 带 [n] 占位符的中文}，拼装时优先于机翻")
    ap.add_argument("--stage", default="all", choices=["all", "mt", "assemble"],
                    help="all=翻译+拼装；assemble=只用缓存重新拼装（改词表后用）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 块（试译）")
    ap.add_argument("--model", default="marian", choices=sorted(MODELS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--list", action="store_true", help="只列出块，不翻译")
    args = ap.parse_args()

    page = Path(args.page).read_text(encoding="utf-8")
    blocks = iter_blocks(page)
    if args.limit:
        blocks = blocks[: args.limit]
    print(f"提取到 {len(blocks)} 个块（段落 + 图注 + 标题）")

    if args.list:
        for i, (tag, _off, inner) in enumerate(blocks):
            plain = html_mod.unescape(re.sub(r"<[^>]+>", "", inner))
            print(f"{i:4d}  {blk_key(inner)}  {tag:10s} {len(inner):6d}B  "
                  f"{re.sub(chr(32) + '+', ' ', plain)[:60]!r}")
        return

    out_path = Path(args.out)
    cache_path = Path(args.cache)
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8")).get("blocks", {})
    elif out_path.exists():                                   # 兼容早期把缓存写进 --out 的版本
        cache = json.loads(out_path.read_text(encoding="utf-8")).get("blocks", {}) or {}
    if isinstance(cache, list):
        cache = {}

    annotated = [(tag, bid, blk_key(inner), *annotate(inner))
                 for bid, (tag, _off, inner) in enumerate(blocks)]

    overrides: dict[str, str] = {}
    ov_atoms: dict[str, str] = {}
    ov_path = Path(args.overrides)
    if ov_path.exists():
        raw_ov = json.loads(ov_path.read_text(encoding="utf-8"))
        ov_atoms = {str(k): v for k, v in raw_ov.pop("_atoms", {}).items()}
        overrides = {str(k): v for k, v in raw_ov.items() if not str(k).startswith("_")}
        print(f"人工校订层：{len(overrides)} 段、{len(ov_atoms)} 个短语")

    # 任务：(块 key, 分块下标 / 原子序号, 待译文本)
    tasks: list[tuple[str, int | str, str]] = []
    chunks_of: dict[int, list[str]] = {}
    for _tag, bid, key, text, atoms in annotated:
        chunks_of[bid] = chunk(text)
        for ci, c in enumerate(chunks_of[bid]):
            tasks.append((key, -1 - ci, c))
        for n, (mode, _raw, inner_text) in enumerate(atoms, start=1):
            if mode == "trans" and inner_text:
                tasks.append((key, n, inner_text))

    def save_cache() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"blocks": cache, "blockList": [b[1] for b in blocks]},
                                         ensure_ascii=False), encoding="utf-8")

    def translate(todo: list[tuple[str, int | str, str]], label: str) -> None:
        tr = Translator(args.model, args.batch)
        step = 40                                            # 每 40 段落一次盘，中断了也能续
        for i in range(0, len(todo), step):
            part = todo[i:i + step]
            for (bid, idx, _src), zh in zip(part, tr([t[2] for t in part])):
                cache[f"{bid}#{idx}"] = zh
            save_cache()
            print(f"  …已保存 {min(i + step, len(todo))}/{len(todo)} 段{label}", flush=True)

    def assemble() -> tuple[list[str], list[int]]:
        """拼装全部块。先用整段译文；占位符丢了就退到「按占位符切段逐段翻」的结果。"""
        final: list[str] = []
        bad: list[int] = []
        for _tag, bid, key, text, atoms in annotated:
            zh_atoms = {}
            for n, (mode, _raw, en_text) in enumerate(atoms, start=1):
                zh_atoms[n] = polish(glossary(en_text) or ov_atoms.get(f"{bid}#{n}")
                                     or cache.get(f"{key}#{n}", ""))
            if str(bid) in overrides:                        # 人工校订优先（overrides 按块下标写）
                block, lost = restore(overrides[str(bid)], atoms, zh_atoms)
                if block:
                    final.append(block)
                    continue
                print(f"  ! 人工校订层的占位符对不上，块 {bid}（缺 {lost}）")
            segs_of_block = split_segments(text)
            # 标题这种「整块就是词典里的一个词」的情况，直接用词典，别让模型去猜
            plain_only = "".join(p if k % 2 else " " for k, p in enumerate(segs_of_block)).strip()
            g = glossary(plain_only)
            if g:
                body = "".join(p if k % 2 else (f" {g} " if p.strip() else p)
                               for k, p in enumerate(segs_of_block))
                block, _lost = restore(polish(body), atoms, zh_atoms)
                if block:
                    final.append(block)
                    continue
            parts = [cache.get(f"{key}#{-1 - ci}") for ci in range(len(chunks_of[bid]))]
            if all(parts):
                block, _lost = restore(polish(" ".join(parts)), atoms, zh_atoms)
                if block:
                    final.append(block)
                    continue
            segs = split_segments(text)                      # 段级回退：标记是自己拼回去的，必定不丢
            if all(k % 2 or cache.get(f"{key}#s{k}") is not None for k, p in enumerate(segs) if p):
                body = "".join(p if k % 2 else polish(cache.get(f"{key}#s{k}", ""))
                               for k, p in enumerate(segs))
                block, _lost = restore(body, atoms, zh_atoms)
                if block:
                    final.append(block)
                    continue
            final.append("")
            bad.append(bid)
        return final, bad

    todo = [t for t in tasks if not cache.get(f"{t[0]}#{t[1]}")]
    print(f"待翻译 {len(todo)} 段，已缓存 {len(tasks) - len(todo)} 段")
    if todo and args.stage != "assemble":
        translate(todo, "正文/短语")
    elif todo:
        print(f"assemble 模式：{len(todo)} 段还没翻过，跳过")

    final, fallback = assemble()

    # 第二遍：整段翻不了（占位符被吃掉）的块，改成按占位符切成小段分别翻，逐段拼回去
    if fallback and args.stage != "assemble":
        seg_tasks: list[tuple[str, int | str, str]] = []
        for _tag, bid, key, text, _atoms in annotated:
            if bid not in fallback:
                continue
            for k, p in enumerate(split_segments(text)):
                if not p or k % 2 or cache.get(f"{key}#s{k}") is not None:
                    continue
                seg_tasks.append((key, f"s{k}", p))
        print(f"整段占位符丢失的块 {len(fallback)} 个 → 改走段级翻译，新增 {len(seg_tasks)} 段")
        if seg_tasks:
            translate(seg_tasks, "段级回退")                 # type: ignore[arg-type]
            final, fallback = assemble()

    out_path.write_text(json.dumps({"blocks": final, "meta": {
        "source": Path(args.page).name,
        "model": MODELS[args.model],
        "count": len(final),
        "fallback": fallback,
        "note": "机翻（本地 MT 模型，逐段对照阅读用）；公式 / 引用 / 图表编号保持原样。",
    }}, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"完成：{out_path}（{len(final)} 块，回退 {len(fallback)} 块：{fallback}）")
    for i, (tag, _off, inner) in enumerate(blocks[:3]):
        print("-" * 70)
        print("[英文]", re.sub(r"<[^>]+>", "", inner)[:150])
        print("[中文]", re.sub(r"<[^>]+>", "", final[i])[:150])


if __name__ == "__main__":
    sys.exit(main())
