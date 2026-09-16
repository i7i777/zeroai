# Grokking 论文中文站（web/）

把 `../docs/` 下的论文（arXiv:2301.05217）译成中文网页，并提供可直接运行的 TypeScript 后端与纯静态前端。

```
web/
├─ server/                后端（TypeScript，Node >= 24 直接运行）
│  ├─ index.ts            静态站点 + /api/meta + /paper.pdf 托管
│  └─ package.json
└─ public/                前端（纯静态：html / css / js / images）
   ├─ index.html          首页
   ├─ paper-zh.html       论文中文翻译（含公式与本地复现图）
   ├─ css/style.css
   ├─ js/main.js          目录高亮 / 阅读进度 / 服务器状态
   └─ images/             复现图（来自 ../python/figures）
```

## 运行

后端无需编译、无需 `npm install`（Node.js ≥ 24 原生支持直接运行 .ts）：

```bash
cd web/server
npm start          # 等效于 node index.ts
# 或直接
node index.ts
```

然后浏览器打开 <http://127.0.0.1:3000>。

## 页面与接口

| 路径 | 说明 |
| --- | --- |
| `/` | 首页 |
| `/paper-zh.html` | 论文中文翻译页 |
| `/paper.pdf` | 原始英文 PDF（由后端从 `../docs/` 托管） |
| `/api/meta` | JSON 元信息（服务器 / 论文 / 目录） |

## 模型可视化页的 LOGITS 面板（`model.html`）

“LOGITS 循环”里第 ③ 块的 2D 热图是**逐点全分辨率**（P×P = 113×113，约 1.3 万次前向、≈4 s，
按 run/seed/epoch 缓存，换输出类或换显示量即时重画），显示量可选：

| 显示量 | 含义 | 外观 |
| --- | --- | --- |
| `softmax 概率 p(c\|a,b)`（默认） | 模型真正把类 c 当作答案的位置 | 只有一条 a+b ≡ c (mod P) 的对角红线，其余为白 |
| 相对 logit `logit_c − mean_c′ logit_c′` | 去掉每条输入的公共分量 | 对角带最红，背景是模型的周期条纹 |
| 原始 logit | 未做任何处理的 `logit_c(a,b)` | 整幅是周期条纹，且多数为正 → 以 0 为白时整体偏红 |

> 注意：**原始 logit 不是“一条对角线”**。模型学到的是稀疏傅里叶解（本仓库 key_freqs 为
> `[5, 10, 15, 31, 48]`），所以固定某个类别的 logit 本身随 a+b 快速振荡，在整平面上形成周期条纹
> ——这与论文图 4 的“周期性”现象一致。要展示“a+b≡c 对角线”应看 softmax 概率或相对 logit。
> 3D 曲面仍是 ~29×29 的粗网格概览，画不出只有 1 格宽的对角线。

## 结构图上的高亮粒度（`model.html`）

结构图不再“整块”高亮，而是尽量落到**层内被分析的那个子元素**：

| 按钮 | 高亮对象 | 声明 |
| --- | --- | --- |
| 🎼 EMBED 频率 | EMBED 里那条 `W_E[a]` 向量（a / b / = 三条之一），不点亮整块 EMBED | `FREQ_SRC.embed.vec = "a"` |
| 🎯 ATTENTION 频率 | ATTENTION 里最下面那条「Σ 4 头输出相加 → W_O 投影」子层（4 个头的 out 通过右侧汇聚线收进它） | `FREQ_SRC.attn.part = "attnSum"` |
| 🧩 MLP 频率 | MLP 最右边那一列**输出向量**（= 加回残差流的 128 维增量，即面板画的那条激活；`W_in`/`W_out` 只是层次与尺寸标注，不会被点亮） | `FREQ_SRC.mlp.part = "mlpOut"` |
| 📊 LOGITS 频率 | 右侧 logits 输出向量里**所选输出类的那一格**（换“输出类”下拉框，结构图上那格会跟着移动） | `FREQ_SRC.logits.part = "logitsCell"` |

映射写在 `js/model.js` 的 `FREQ_SRC[*].vec / .part` 里，由 `highlightArch()` 读取（`g[data-vec]` / `g[data-part]` 上加 `.on`）。

## 读出端：UNEMBED 矩阵（权重）与 logits（输出）

结构图把这两块并排画、**行高对齐**：第 r 行 `W_U[r]` 和第 r 格 `logit` 是同一个输出类（金色 = 真值类）。

| | UNEMBED | logits |
| --- | --- | --- |
| 画的是什么 | 读出**权重矩阵** `W_U`（113 类 × 128 维） | 读出**结果向量**（113 个 logit） |
| 类型 | 静态权重，与输入无关 | 激活，= `W_U · x`，取决于输入 a、b |
| 有没有面板 | ❌ 无（原「🔓 UNEMBED 频率」已下线，它看的是权重） | ✅ 📊 LOGITS 频率 |
| 论文位置 | §3.1 / 图 3（右）的 `W_L` 系列分析 | §4.1 / 图 4（右）的 logits 周期性 |

> 权重视图删掉不影响论文图 3 那条线：下方热图区已有 `W_E`、`W_L = W_U·W_out` 的热图与“傅里叶范数”面板。

## 4 个频率面板各画的是什么（权重 vs 激活）

见 `js/model.js` 的 `freqSignal()`：

| 按钮 | 取数来源 | 类型 | 需要前向吗 |
| --- | --- | --- | --- |
| 🎼 EMBED 频率 | `token_embed[a][c]`，a = 0…P−1 | **权重矩阵** W_E（第 a 行就是 token a 的嵌入向量） | 否，直接读 |
| 🎯 ATTENTION 频率 | `forward(a=0, b).attn[c]`，b = 0…P−1 | **激活**（位置「=」4 头输出之和 = 残差流增量） | 是，113 次前向 |
| 🧩 MLP 频率 | `forward(a=0, b).mlp[c]`，b = 0…P−1 | **激活**（MLP 输出增量 = W_out·ReLU(…) + b_out） | 是，113 次前向 |
| 📊 LOGITS 频率 | `forward(a=0, b).logits[c]`，b = 0…P−1 | **输出** logits | 是，113 次前向 |

EMBED 是唯一的“权重”面板：`W_E` 的行就是 token 的嵌入向量，它同时是喂进网络的激活，所以并不矛盾。
ATTENTION / MLP / LOGITS 都是激活或输出，它们只有在给定输入时才存在，所以面板固定 a = 0、扫 b 跑一遍前向再对 b 做 DFT。

> 注：`a` 在这里只是个「曲线自变量」的名字：EMBED 面板扫的是 **token 轴**（不是某次输入的 a），
> ATTENTION / MLP / LOGITS 面板扫的是 **位置 2 的 token b**（位置 1 固定为 a = 0）。

## 说明

- MLP 权重形状容易看反：图上按 **PyTorch 的存法**标注，`nn.Linear(in, out).weight.shape = [out, in]`。
  所以 `W_in`（`mlp_in = nn.Linear(128, 512)`）是 **512×128**，`W_out`（`mlp_out = nn.Linear(512, 128)`）是 **128×512**。
  自检：`W_L = W_U · W_out` = [113×128]·[128×512] = [113×512] 才成立（见 `python/grokking/model.py:172`）。
- 中文翻译为学习用途的整理，数学排版使用 KaTeX（CDN）。
- 图 2–8 均来自本工作区 `python/` 目录的 PyTorch 复现实验（5 seeds，P=113）。
- 后端仅依赖 Node 内置模块；改端口可用环境变量 `PORT`（默认 3000）。
