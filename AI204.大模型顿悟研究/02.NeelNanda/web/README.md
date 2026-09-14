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

## 说明

- 中文翻译为学习用途的整理，数学排版使用 KaTeX（CDN）。
- 图 2–8 均来自本工作区 `python/` 目录的 PyTorch 复现实验（5 seeds，P=113）。
- 后端仅依赖 Node 内置模块；改端口可用环境变量 `PORT`（默认 3000）。
