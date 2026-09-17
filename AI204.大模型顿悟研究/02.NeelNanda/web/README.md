# Grokking 论文研读站（web/）

站点围绕论文（arXiv:2301.05217）搭建：**首页就是论文英文原文的本地完整版**，另附本地复现的模型可视化页。后端是可直跑的 TypeScript，前端是纯静态页面。

```
web/
├─ server/                后端（TypeScript，Node >= 24 直接运行）
│  ├─ index.ts            静态站点 + /api/meta + /paper.pdf 托管
│  └─ package.json
├─ tools/                 构建脚本
│  ├─ localize_arxiv_html.py  arXiv/ar5iv 的 LaTeXML 渲染 → public/index.html + paper-en-local.html（本地完整版）
│  └─ pdf2html.py          docs/*.pdf → public/paper-en-pdf.html（备用：从 PDF 抠字重排，依赖 PyMuPDF）
└─ public/                前端（纯静态：html / css / js / images / assets / viz-data）
   ├─ index.html          **站点首页 = 论文英文原文（本地完整版）**，由 localize_arxiv_html.py 生成，勿手改
   ├─ paper-en-local.html 内容与首页完全相同（保留给旧链接/书签）
   ├─ model.html          模型可视化（多 seed 对比 / 准确率曲线 / 结构图 / 测试 / 频率面板）
   ├─ css/style.css
   ├─ js/main.js          目录高亮 / 阅读进度 / 服务器状态
   ├─ js/model.js         模型可视化页逻辑
   ├─ images/             复现图（来自 ../python/figures）
   └─ assets/
      ├─ ar5iv/           LaTeXML 样式表（3 个 CSS）与站点 logo
      └─ paper-en/        英文原版插图（18 张，含 figs/ 子目录）
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
| `/` | **首页 = 论文英文原文（本地完整版）**：`tools/localize_arxiv_html.py` 把 ar5iv 的 LaTeXML 渲染（正文 DOM + 3 个样式表 + 18 张插图）镜像到本机，离线可读；与 `/paper-en-local.html` 内容相同 |
| `/paper-en-local.html` | 同首页（保留给旧链接） |
| `/model.html` | 模型可视化页 |
| `/paper.pdf` | 原始英文 PDF（由后端从 `../docs/` 托管；站内已无入口链接，直接访问仍可用） |
| `/api/meta` | JSON 元信息（服务器 / 论文 / 目录） |

> 英文正文**没有被手工抄写或改写**进本仓库。排版正确的英文源见
> **arXiv 官方 HTML 版**：<https://arxiv.org/html/2301.05217v3>（该页无 `X-Frame-Options` / `frame-ancestors` 限制，可直接内嵌；
> ar5iv 与 abs 页是 `SAMEORIGIN`，只能外链）；首页则是 `tools/localize_arxiv_html.py`
> **镜像 ar5iv 的渲染结果**到本地，离线可读。两者都是 LaTeXML 编译产物，公式（MathML）/图表/章节是真排版；
> 页顶页底都标了原始出处。

## 重新生成首页（论文本地完整版）

```bash
# 无第三方依赖，只用标准库；默认从 ar5iv 抓取，同时写 public/index.html 与 public/paper-en-local.html
python web/tools/localize_arxiv_html.py
# 可选：--url 换源（比如 https://arxiv.org/html/2301.05217v3）、--out 只写指定的一份
```

脚本做的事：抓取源页面 → 只取 `<body>` 正文 DOM（丢掉原始 head 元信息）→ 下载 3 个 LaTeXML 样式表
到 `public/assets/ar5iv/`、下载 18 张插图到 `public/assets/paper-en/`（保留 `figs/` 子目录）→
把页面里的资源路径改写成站内绝对路径 → 修掉 LaTeXML 0.7.5 把 `\left(`/`\)` 渲染成字面量 `OPEN`/`CLOSE` 的旧毛病
→ 加上本站顶部导航条、出处说明与页脚。生成的 HTML **不要手工改**，改脚本再跑一遍。

## 备用脚本：纯从 PDF 抠字重排（`paper-en-pdf.html`）

在线渲染源都不可用时的应急方案，产物不在站点导航里：

```bash
# 需要 PyMuPDF；默认读 ../docs/*.pdf，写 public/paper-en-pdf.html + public/images/en/*.png
python web/tools/pdf2html.py
# 可选：--pdf 指定 PDF、--out 指定输出、--zoom 3 控制插图裁剪倍率
```

脚本做的事：PDF 逐页取文字块与矢量/位图对象 → 聚类出插图与表格区域（裁剪成 PNG，216 dpi）
→ 余下文字按字号/字体/全大写等特征分成章标题（12 pt）、小节标题（10 pt 全大写）、题注（`Figure N:`）、
独立公式（数学字体占比 > 50%）与正文段落 → 跨块续段与行尾断词自动合并 → 写成站点风格的 HTML。
只丢弃页眉/页码/水印与图内文字（`pdf2html.py --help` 与脚本头部注释里有完整说明）。

## 模型可视化页（`model.html`）
页面顶部是「⓪ 各 seed 横向比较」卡片，一行一格、共 5 格，顶部有两个视角切换：

* **📈 效果：准确率曲线** —— 每格画该 seed 的训练/测试准确率曲线（绿虚线 = 顿悟点，金虚线只画在当前选中的格子里），
  格下标注该 seed 的**顿悟 epoch** 与最终测试准确率；数据来自 `index.json` 的 `curve`，**零下载**。
* **🧬 参数：W<sub>E</sub> 关键频率** —— 每格画该 seed 在**当前 epoch** 的 W<sub>E</sub> 频率谱
  （对 token 轴 DFT，柱高 = Σ<sub>c</sub>|DFT<sub>c</sub>(k)|²，按格归一），格下标注 top-3 频率与能量占比。
  需要逐个 seed 取权重快照（887 KB / 个），结果按 `(run|seed|epoch)` 缓存；
  各 seed 导出的快照不一定一样（本 run 里 seed 3/4 只有最终一档），缺失时会退到该 seed 最近的快照并标注「最近快照」。

点击任意一格即切到该 seed（与工具栏 Seed 下拉、下方 ① 卡片走同一条 `onSeed()` 路径）；格数随 `index.json` 里的 seed 数自动增减。

### 这 5 个 seed 到底差在哪（`python/_seed_diff_probe*.py` 实测）

超参数完全相同（`p=113, d_model=128, n_heads=4, d_head=32, n_mlp=512, frac_train=0.3, λ=1, lr=1e-3`），
差异只有两处，都由 `seed` 带入：

1. **训练/测试划分**：`make_modular_addition_data(p, frac, seed)` 用 `np.random.default_rng(seed)` 抽 30% 作训练集
   （3831 / 12769 组）。两个 seed 的训练集只重叠 **≈1150 组（~30%）**，即约 70% 的训练样本不同。
2. **随机初始化**：`torch.manual_seed(seed)` 决定权重初值。

后果（在最终快照上实测）：

* 每个 seed 都学出了**稀疏傅里叶电路**：W<sub>E</sub> 的能量 91–96% 集中在 6 个频率上，输出端 93–97%；
* 但**用哪几个频率完全不同**：W<sub>E</sub> 的 top-6 频率两两只重合 0–1 个，输出端 0–2 个，**5 个 seed 没有一个共同频率**；
* 顿悟 epoch 差很多（2551 / 4051 / 7126 / 3476 / 9626），但最终训练/测试准确率都是 100%。

一句话：**算法相同（角度表示 + 三角恒等式），实现不同（选了哪组频率、哪组维度）** ——
这正是表征的旋转/置换自由度 + 频率选择自由度。

紧接其下是「① 准确率曲线」卡片，画的是**当前选中的实验 / seed** 的准确率曲线：

* 蓝线 = 训练集，红线 = 测试集；绿色虚线 = 顿悟点（test acc 首次 ≥ 95% 的 epoch），金色虚线 = 当前查看的 epoch；
* 曲线随工具栏的「实验 / Seed」下拉与「阶段(epoch)」滑块联动重画；
* 点击或拖动曲线即可切换阶段（按 x 坐标换算成 epoch，再吸附到最近的权重快照），与滑块、结构图高亮同一状态；
* 图例里实时显示 `P / d_model / heads / frac_train`、最终准确率、顿悟点，以及当前 epoch 的训练/测试准确率与所处阶段；
* 曲线上的竖直回落是训练中的 **loss spike**（个别 epoch 短暂退化）——数据原样画出，不做平滑。

再往下是「② 网络结构」（`#archSvg`）与说明/测试按钮条 —— 页面到此也是最后一块，
**原先下方的权重面板区（`W_E` / `W_L` / 注意力头 / MLP 权重热图与「训练曲线」）已整块删除**；
要用到的只有 `state.mats`（前向计算用），所以 `renderHeat` / `renderBars` / `matView()` / `drawCurve()`
这四个函数也一并删掉了，`model.js` 里不再有任何权重热图代码。
工具栏右侧那个「权重已载入 eNNNNN」是**状态显示**（不是按钮）：权重在切换 Seed、拖动 epoch、点击曲线时都会自动载入。

### 「📊 LOGITS 频率」面板的三块图

面板固定 `a = 0`、扫 `b = 0…P−1`（位置 1 固定，位置 2 做自变量），对所选输出类的 logit 做分析：

| 块 | 画的是什么 |
| --- | --- |
| ① 波形 | 蓝/灰 = 原始 `logit_c(a=0, b)`，橙 = **只用幅度最大的前 N 个频率**重建出来的曲线；标题给出重建的复现度（例：3 个频率复现 86%） |
| ② 频率谱 | 对 `b` 做 DFT 后各频率 k 的幅度（红 = 被保留的 N 个），标题给出它们占总能量的比例 |
| ③ 拆解与叠加 | 上半 = 把保留的频率**一条条加上去**（浅 → 深，最后一条 = 重建曲线）；下半 = 每条余弦波 `R·cos(2πkb/P − φ)` 单独的样子，幅度/相位完全由权重决定 |

> 论文里的“周期性”就是这块：网络学到稀疏傅里叶解，logit 只在少数频率上有分量
> （本 run seed 0 的关键频率 `k = 5, 10, 15, 17, 20, 48`），几个余弦波相长/相消，
> 正确类别 `c* = a + b` 处被抬到最大。换「输出类 / 维度」下拉或换 epoch 可对比。
> 面板**没有** 2D/3D 曲面 —— 想看 5 个 seed 的关键频率差异用 ⓪ 卡片的「🧬 参数」视图。

## 结构图里画的是「真实输出张量」

结构图的 EMBED / ATTENTION / MLP / LOGITS 四个框里**不再是示意图**，而是用
**当前阶段的权重 × 当前输入样本** 现场跑一次前向（`archTensors()` → `forwardCore()`，
与 PyTorch 的 `forward_full()` 完全一致），把各模块的**输出张量**画成条带：

| 框 | 画的内容（张量） | 形状 |
| --- | --- | --- |
| EMBED | 3 个位置各自的 `W_E[token] + pos` | 3 × 128 |
| ATTENTION | 每个头 `W_O·(attn·V)` 的输出，加一条「Σ 4 头」 | 4 × 128，再加 128 |
| MLP | 输入 `resid2` / 隐层 `ReLU` 激活 / 输出 `W_out·h + b_out` | 128 / 512 / 128 |
| LOGITS | 113 个类的 logit（条长 ∝ logit） | 113 |

画法约定：

- 每根细竖条 = 张量的一个分量，以**中线为基准**（琥珀向上 = 正，天蓝向下 = 负），
  高度 ∝ `(|数值| / 该条带最大值)^0.65` —— 幂次压缩是为了让重尾分布里的小分量也看得见；
- MLP 三层**各自归一**，真实的幅度差别写在右侧的 `max|·| = …` 上
  （seed 0 / e25000 / a=37,b=70：`resid2` 3.54、隐层 34.19、输出 278.62，与参考实现逐位对上）；
- LOGITS 每格底层的深底色 + 按相对大小着色的条带，青绿色 = argmax（预测类），金框 = 真值类；
- 重算时机：切换 Seed（旧权重立即作废，先显示“权重载入中…”）、拖动 epoch / 点曲线、点「测试」换样本，
  都在 `loadWeights()` 成功或 `runTest()` 之后调 `buildArch()` 重画。

## 结构图上的高亮粒度（`model.html`）

结构图不再“整块”高亮，而是尽量落到**层内被分析的那个子元素**：

| 按钮 | 高亮对象 | 声明 |
| --- | --- | --- |
| 🎼 EMBED 频率 | EMBED 里那条 `W_E[a]` 向量（a / b / = 三条之一），不点亮整块 EMBED | `FREQ_SRC.embed.vec = "a"` |
| 🎯 ATTENTION 频率 | ATTENTION 里最下面那条「Σ 4 头输出相加 → W_O 投影」子层（4 个头的输出通过右侧汇聚线收进它） | `FREQ_SRC.attn.part = "attnSum"` |
| 🧩 MLP 频率 | MLP 框里第三条**输出条带**（`W_out·ReLU(…) + b_out`，即加回残差流的 128 维增量，也是面板画的那条激活） | `FREQ_SRC.mlp.part = "mlpOut"` |
| 📊 LOGITS 频率 | 右侧 logits 里**所选输出类的那一格**（换“输出类”下拉框，结构图上那格会跟着移动） | `FREQ_SRC.logits.part = "logitsCell"` |

映射写在 `js/model.js` 的 `FREQ_SRC[*].vec / .part` 里，由 `highlightArch()` 读取（`g[data-vec]` / `g[data-part]` 上加 `.on`）。

## 读出端：结构图上只画 logits（UNEMBED 块已删）

结构图现在的链路是 **输入 → EMBED → ATTENTION → MLP → logits**，最后一格写的是
`logits`（副标题 `W_U·x → 113 类`）：113 个矩形格子 = 113 个输出类的 logit，
条长 ∝ 该类的 logit 大小，青绿 = argmax（预测类），金色框 = 真值类，测试后还会在下方标出 `ĉ = …`。

原来在 MLP 与 logits 之间还画过一块 **UNEMBED**（`W_U` 权重矩阵，113 类 × 128 维，行高与 logits 对齐），
已按需求删掉：它只是静态权重，真正的信息在读出结果里。想分析 `W_U` 的频率信息可以用：

| 想看的东西 | 去哪里看 |
| --- | --- |
| 读出的结果（113 个 logit）随输入怎么变 | 📊 LOGITS 频率面板（① 波形+重建 / ② 频率谱 / ③ 拆解叠加） |
| 5 个 seed 学到的关键频率差异 | ⓪ 卡片的「🧬 参数：W_E 关键频率」视图 |
| `W_U · W_out = W_L` 这类权重组合 | 代码里 `freqSignal()` / 测试面板的 `state.mats.unembed`（页面上不再出热图） |

> 论文里对应 §3.1 / 图 3（`W_L` 系列）与 §4.1 / 图 4（logits 周期性）。

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

- MLP 权重形状容易看反：图上按 **PyTorch 的存法**标注（写在 MLP 框底部），`nn.Linear(in, out).weight.shape = [out, in]`。
  所以 `W_in`（`mlp_in = nn.Linear(128, 512)`）是 **512×128**，`W_out`（`mlp_out = nn.Linear(512, 128)`）是 **128×512**。
  自检：`W_L = W_U · W_out` = [113×128]·[128×512] = [113×512] 才成立（见 `python/grokking/model.py:172`）。
- 首页（英文原文镜像）的正文版权归原作者，按 arXiv 非独占许可发布，仅供本地研读。
- 图 2–8 均来自本工作区 `python/` 目录的 PyTorch 复现实验（5 seeds，P=113）。
- 后端仅依赖 Node 内置模块；改端口可用环境变量 `PORT`（默认 3000）。
