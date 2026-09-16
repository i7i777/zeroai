/* 模型可视化 —— 网络结构 + 按 seed/阶段加载权重做研究
 * 数据来源: python/grokking/vizexport.py 导出的 /viz-data/**（.bin 快照 + index.json）
 */
(() => {
  "use strict";

  // ---------------- 小工具 ----------------
  const $ = (s, el) => (el || document).querySelector(s);
  const pad6 = (n) => String(n).padStart(6, "0");

  // ---------------- 快照解析 ----------------
  async function fetchJSON(u) {
    const r = await fetch(u);
    if (!r.ok) throw new Error(`${u} -> ${r.status}`);
    return r.json();
  }

  /** 解析 .bin：GKV1 魔数 + uint32 头长 + JSON 头 + float32 权重 */
  function parseBin(buf) {
    const dv = new DataView(buf);
    if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== "GKV1") throw new Error("bad magic");
    const headLen = dv.getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headLen)));
    // 从头部结束处截取权重区（slice 保证新缓冲区按 4 字节对齐）
    const f32 = new Float32Array(buf.slice(8 + headLen));
    const mats = {};
    for (const m of header.mats) {
      mats[m.name] = {
        shape: m.shape,
        data: f32.slice(m.start, m.start + m.numel),
      };
    }
    return { header, mats };
  }

  // ---------------- 颜色 ----------------
  // 发散色：蓝-白-红（用于权重热图，0 为白）
  function divergingRGB(t) {
    // t in [0,1]
    const stops = [[33, 102, 172], [255, 255, 255], [178, 24, 43]];
    const seg = t < 0.5 ? 0 : 1;
    const local = t < 0.5 ? t * 2 : (t - 0.5) * 2;
    const a = stops[seg], b = stops[seg + 1];
    return [
      Math.round(a[0] + (b[0] - a[0]) * local),
      Math.round(a[1] + (b[1] - a[1]) * local),
      Math.round(a[2] + (b[2] - a[2]) * local),
    ];
  }
  // 单色深→亮（用于范数/谱）
  function monoRGB(t) {
    const c = Math.round(20 + 235 * Math.pow(t, 0.55));
    return [c, Math.round(40 + 160 * t), Math.round(30 + 120 * t)];
  }

  /** 在 <canvas> 上画矩阵热图
   * @param shape [srcRows, srcCols]
   * @param get(i) 返回按行优先的第 i 个源值
   * @param transpose true 表示显示时把源矩阵转置（用源列作 y、源行作 x）
   */
  function renderHeat(canvas, shape, get, { transpose = false, symmetric = true, norm = false } = {}) {
    const srcRows = shape[0], srcCols = shape[1];
    const rowsD = transpose ? srcCols : srcRows;   // 显示高度方向像素数
    const colsD = transpose ? srcRows : srcCols;   // 显示宽度方向像素数
    const maxDimPx = 320, maxWidePx = 820;
    const px = Math.max(1, Math.min(
      Math.floor(maxDimPx / rowsD),
      Math.floor(maxWidePx / colsD),
      4
    ));
    const W = colsD * px, H = rowsD * px;
    canvas.width = W; canvas.height = H;
    canvas.style.maxWidth = "100%"; canvas.style.height = "auto";

    // 数值范围（采样）
    let lo = Infinity, hi = -Infinity;
    const nSrc = srcRows * srcCols;
    const step = Math.max(1, Math.floor(nSrc / 60000));
    for (let i = 0; i < nSrc; i += step) {
      const v = get(i);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (symmetric) { const a = Math.max(Math.abs(lo), Math.abs(hi)); lo = -a; hi = a; }
    if (lo === hi) hi = lo + 1e-9;

    const img = canvas.getContext("2d").createImageData(W, H);
    const d = img.data;
    for (let y = 0; y < H; y++) {
      const iy = Math.floor(y / px);           // 显示行
      for (let x = 0; x < W; x++) {
        const ix = Math.floor(x / px);         // 显示列
        // 源位置
        const sr = transpose ? ix : iy;
        const sc = transpose ? iy : ix;
        const v = get(sr * srcCols + sc);
        const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
        const rgb = norm ? monoRGB(t) : divergingRGB(t);
        const o = (y * W + x) * 4;
        d[o] = rgb[0]; d[o + 1] = rgb[1]; d[o + 2] = rgb[2]; d[o + 3] = 255;
      }
    }
    canvas.getContext("2d").putImageData(img, 0, 0);
  }

  /** 在 <canvas> 上画一维柱状图（傅里叶范数等） */
  function renderBars(canvas, values, labels) {
    const W = 920, H = 240;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const n = values.length;
    const pad = 4, baseY = H - 24, topY = 18;
    const maxV = Math.max(...values, 1e-12);
    const bw = (W - pad * 2) / n;
    ctx.fillStyle = "steelblue";
    for (let i = 0; i < n; i++) {
      const h = ((H - topY - (H - baseY)) * values[i]) / maxV;
      ctx.fillRect(pad + i * bw, baseY - h, Math.max(1, bw - 1), h);
    }
    ctx.fillStyle = "#57606a";
    ctx.font = "11px sans-serif";
    if (labels && labels.length === n) {
      const every = Math.max(1, Math.ceil(n / 14));
      ctx.textAlign = "center";
      for (let i = 0; i < n; i += every) {
        ctx.fillText(String(labels[i]), pad + i * bw + bw / 2, H - 6);
      }
    }
    ctx.textAlign = "left";
    ctx.fillStyle = "#888";
    ctx.fillText(`0  ~  ${maxV.toExponential(2)}`, 6, topY - 2);
  }

  // ---------------- 傅里叶（朴素实 DFT） ----------------
  function buildTwiddles(p) {
    // tw[k][n] = exp(-2πi k n / p)
    const tw = [];
    for (let k = 0; k < p; k++) {
      const row = [];
      for (let n = 0; n < p; n++) {
        const a = (-2 * Math.PI * k * n) / p;
        row.push([Math.cos(a), Math.sin(a)]);
      }
      tw.push(row);
    }
    return tw;
  }

  /** 对矩阵 [rows=p, cols] 求沿行方向的实谱范数：norm[k]=sqrt(sum_c|DFT_c(k)|^2)，k=0..p/2 */
  function fourierNorms1d(data, p, cols) {
    const tw = buildTwiddles(p);
    const kMax = Math.floor(p / 2);
    const norms = new Array(kMax + 1).fill(0);
    for (let k = 0; k <= kMax; k++) {
      let sum = 0;
      for (let c = 0; c < cols; c++) {
        let re = 0, im = 0;
        const base = c;
        for (let n = 0; n < p; n++) {
          const v = data[n * cols + c];
          re += v * tw[k][n][0];
          im += v * tw[k][n][1];
        }
        sum += re * re + im * im;
      }
      norms[k] = Math.sqrt(sum);
    }
    return norms;
  }

  /** 一维实信号的傅里叶级数（P 为素数时与 DFT 完全等价）：
   *    y[n] ≈ c0 + Σ_k [ a_k cos(2πkn/P) + b_k sin(2πkn/P) ]，k = 1…⌊P/2⌋
   *  返回 { c0, comps:[{k,a,b,amp}], K, energy }（energy = 去掉直流后的总能量）。
   */
  function fourierSeries(y, p) {
    let c0 = 0;
    for (let n = 0; n < p; n++) c0 += y[n];
    c0 /= p;
    let energy = 0;
    for (let n = 0; n < p; n++) energy += (y[n] - c0) * (y[n] - c0);

    const K = Math.floor(p / 2);
    const comps = [];
    for (let k = 1; k <= K; k++) {
      let a = 0, b = 0;
      for (let n = 0; n < p; n++) {
        const th = (2 * Math.PI * k * n) / p;
        a += y[n] * Math.cos(th);
        b += y[n] * Math.sin(th);
      }
      a *= 2 / p; b *= 2 / p;
      comps.push({ k, a, b, amp: Math.hypot(a, b) });
    }
    return { c0, comps, K, energy };
  }

  /** 只用 keep 里的频率重建信号（直流分量恒保留）。 */
  function rebuildFromFreqs(series, p, keep) {
    const out = new Float64Array(p).fill(series.c0);
    for (const c of series.comps) {
      if (!keep.has(c.k)) continue;
      for (let n = 0; n < p; n++) {
        const th = (2 * Math.PI * c.k * n) / p;
        out[n] += c.a * Math.cos(th) + c.b * Math.sin(th);
      }
    }
    return out;
  }

  /** 幅度最大的前 N 个频率（按 k 升序返回）。 */
  function topFreqs(series, N) {
    return series.comps.slice().sort((x, z) => z.amp - x.amp).slice(0, N).sort((x, z) => x.k - z.k);
  }

  // ===================================================================
  // 全局状态
  // ===================================================================
  const state = {
    index: null,      // /viz-data/index.json
    run: null, seed: null,
    idx: null,        // 当前 (run,seed) 的 index（config/curve/epochs）
    epochs: [],
    cfg: null,
    mats: null,
    loadedEpoch: null,
    sample: null,      // 测试样本 {a,b,c,pred,correct}（用于在结构图中点亮）
    testList: null,    // 该 (run,seed) 的测试集样本（test_pairs.json）
    testKey: null,
    lastTest: null,    // 最近一次测试的 {a,b,c,src}（切阶段后用它按新权重重算）
    gen: 0,            // 阶段选择序号，用于丢弃过期的异步刷新
    freqSrc: null,     // 当前打开的频率视图：embed/attn/mlp/unembed/logits
    freqPick: {},      // 每级自动挑中的“最正弦”维度/类别（换权重后清空）
    archStage: null,   // 结构图上高亮的级（与按钮/面板同步）
  };

  // ===================================================================
  // 网络结构 SVG（类似论文/示意图风格）
  // ===================================================================
  function buildArch(cfg, sample) {
    const svg = $("#archSvg");
    const p = cfg ? cfg.p : 113;
    const d = cfg ? cfg.d_model : 128;
    const h = cfg ? cfg.n_heads : 4;
    const m = cfg ? cfg.n_mlp : 512;
    const H = 620;                         // 紧凑画布：各模块处于同一横带
    const VS = 0;                          // 取消纵向平移（MLP 与 ATTENTION 等高、同顶对齐）

    // ---- 横向等距布局：5 个部件，块间空隙固定 ----
    const GAP = 64;
    const X0 = 20;
    const wIn = 210, wEmb = 140, wAtt = 270, wMlp = 330, wUn = 130, wLog = 76;
    const inX = X0;
    const embX = inX + wIn + GAP;
    const attX = embX + wEmb + GAP;
    const mlpX = attX + wAtt + GAP;
    const unX = mlpX + wMlp + GAP;
    const logX = unX + wUn + GAP;
    const inCx = inX + wIn / 2, embCx = embX + wEmb / 2, attCx = attX + wAtt / 2,
          mlpCx = mlpX + wMlp / 2, unCx = unX + wUn / 2, logCx = logX + wLog / 2;
    const AY = 310;                        // 箭头/数据流所在的水平线
    const W = logX + wLog + 40;            // 视口总宽（自动适配）

    let s = `<defs>
      <marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
        <path d="M0,0 L9,4.5 L0,9 Z" fill="#8b2f2f"/>
      </marker>
      <marker id="arrL" markerWidth="9" markerHeight="9" refX="1" refY="4.5" orient="auto">
        <path d="M9,0 L0,4.5 L9,9 Z" fill="#c3a1ee"/>
      </marker>
      <linearGradient id="bx" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stop-color="#2b2f3a"/><stop offset="1" stop-color="#3a3f52"/>
      </linearGradient>
    </defs>`;
    const lbl = (x, y, txt, size = 12, fill = "#9aa3b2") =>
      `<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" text-anchor="middle" font-family="sans-serif">${txt}</text>`;

    // —— 输入 / EMBED / ATTENTION / UNEMBED 整体下移（垂直居中的“流水线”上半区）——
    s += `<g transform="translate(0,${VS})">`;

    // 箭头（每段等长 GAP）
    const arrows = [[inX + wIn, embX], [embX + wEmb, attX], [attX + wAtt, mlpX], [mlpX + wMlp, unX], [unX + wUn, logX]];
    for (const [x1, x2] of arrows) {
      s += `<line x1="${x1}" y1="${AY}" x2="${x2}" y2="${AY}" stroke="#8b2f2f" stroke-width="2.4" marker-end="url(#arr)"/>`;
    }

    // 1) 输入：3 个 one-hot token（位置 a、b、=），并标出 a、b 代表的数值
    const NTOK = p + 1;                         // 词表：0..112 与 '='
    const hasS = !!sample;
    const aVal = hasS ? sample.a : Math.max(1, Math.floor(p * 0.33));
    const bVal = hasS ? sample.b : Math.max(1, Math.floor(p * 0.62));
    const tokVals = [aVal, bVal, p];            // 第三列为 '=' token
    const tokTags = [
      hasS ? `位置1 · a=${aVal}  ●测试` : `位置1 · a=${aVal}`,
      hasS ? `位置2 · b=${bVal}  ●测试` : `位置2 · b=${bVal}`,
      `位置3 · '='`,
    ];
    const padX = 10, cw = 48;
    const gapCol = (wIn - 2 * padX - 3 * cw) / 2;
    const IO_TOP = 152, IO_BOT = 450;
    const cellH = (IO_BOT - IO_TOP) / NTOK;
    s += `<g data-stage="input"><rect x="${inX}" y="104" width="${wIn}" height="416" rx="8" fill="url(#bx)" stroke="#5b6172"/>`;
    s += lbl(inX + wIn / 2, 96, "输入 tokens：3 个 one-hot", 12.5, "#dfe3ea");
    for (let t = 0; t < 3; t++) {
      const cx = inX + padX + cw / 2 + t * (cw + gapCol);
      for (let r = 0; r < NTOK; r++) {
        const active = (r === tokVals[t]);
        const y = IO_TOP + r * cellH;
        const col = active ? "#ffb020" : "#273149";
        const stk = active ? ' stroke="#ffe08a" stroke-width="1"' : "";
        s += `<rect x="${cx - cw / 2 + 5}" y="${(y + 0.4).toFixed(1)}" width="${cw - 10}" height="${Math.max(1, cellH - 0.8).toFixed(2)}" rx="1" fill="${col}"${stk}/>`;
      }
      const ay = IO_TOP + tokVals[t] * cellH;
      s += `<rect x="${cx - cw / 2 + 5}" y="${(ay - 0.4).toFixed(1)}" width="${cw - 10}" height="${(cellH + 1).toFixed(2)}" fill="none" stroke="#ffd57a" stroke-width="1.3"/>`;
      s += `<text x="${cx}" y="${IO_BOT + 18}" fill="#dfe3ea" font-size="10.5" text-anchor="middle">${tokTags[t]}</text>`;
      s += `<text x="${cx + cw / 2 + 4}" y="${(ay + cellH / 2 + 3).toFixed(0)}" fill="#ffd57a" font-size="9" font-family="monospace">${tokVals[t]}</text>`;
    }
    const inCap = hasS
      ? `one-hot 长度 ${NTOK} · 测试样本：a=${aVal} b=${bVal}，真值 c=${(aVal + bVal) % p}`
      : `one-hot 长度 ${NTOK}（0…${p - 1} 与 “=”）；共 3 个位置`;
    s += lbl(inX + wIn / 2, IO_BOT + 42, inCap, 10, hasS ? "#ffd57a" : "#9aa3b2");
    s += `</g>`;

    // 2) EMBED：3 个 one-hot（a / b / =）各查一行 W_E，得到 3 个 d 维向量
    //    每个向量单独成组（data-vec），便于高亮到「某一个 token 的嵌入向量」而不是整块 EMBED
    s += `<g data-stage="embed"><rect x="${embX}" y="150" width="${wEmb}" height="320" rx="10" fill="url(#bx)" stroke="#c0843c"/>`;
    s += lbl(embCx, 140, "EMBED", 13, "#c9a15c");
    s += `<text x="${embCx}" y="180" fill="#dfe3ea" font-size="14" text-anchor="middle" font-family="monospace">W_E</text>`;
    const embVecs = [
      { key: "a", val: tokVals[0], col: "#ffd57a" },   // 位置1 的 token a
      { key: "b", val: tokVals[1], col: "#9db4d0" },   // 位置2 的 token b
      { key: "=", val: tokVals[2], col: "#9db4d0" },   // 位置3 的 '='
    ];
    const vX = embX + 18, vW = wEmb - 36, vH = 56, vTop = 200, vGap = 22, nBar = 12;
    for (let i = 0; i < embVecs.length; i++) {
      const v = embVecs[i], vy = vTop + i * (vH + vGap);
      s += `<g data-vec="${v.key}">`;
      s += `<rect x="${vX}" y="${vy}" width="${vW}" height="${vH}" rx="7" fill="#2a3145" stroke="#6b7385"/>`;
      const bw = (vW - 16) / nBar;
      for (let j = 0; j < nBar; j++) {                  // 小竖条 = 该向量 d 个分量的示意
        const op = (0.22 + 0.68 * Math.abs(Math.sin(i * 2.1 + j * 1.7))).toFixed(2);
        s += `<rect x="${(vX + 8 + j * bw).toFixed(1)}" y="${vy + 9}" width="${(bw - 2).toFixed(1)}" height="20" rx="1" fill="${v.col}" fill-opacity="${op}"/>`;
      }
      // 两行文字：名字左对齐、token 值右对齐，避免同一行挤在一起
      s += `<text class="vecName" x="${vX + 8}" y="${vy + vH - 11}" fill="${v.col}" font-size="10.5" font-family="monospace">W_E[${v.key}]</text>`;
      s += `<text x="${vX + vW - 8}" y="${vy + vH - 11}" fill="#9aa3b2" font-size="9.5" text-anchor="end" font-family="monospace">token ${v.val}</text>`;
      s += `</g>`;
    }
    s += lbl(embCx, 452, `one-hot(${NTOK}) → ${d} 维`, 10.5);
    s += `</g>`;

    // 3) ATTENTION（4 头）
    //    头内部：Q/K/V → softmax(QKᵀ/√d)·V → out；4 个头的 out 再由右侧汇聚线收到 Σ 框。
    //    最下面那个「Σ 4 头输出相加 → W_O 投影」才是频率视图分析的那一层（位置「=」的残差流增量）。
    s += `<g data-stage="attn"><rect x="${attX}" y="70" width="${wAtt}" height="480" rx="12" fill="#1f2530" stroke="#7a5bb0"/>`;
    s += lbl(attCx, 55, `ATTENTION（${h} 头）`, 14, "#b18be0");
    const L = attX + 22;
    const HW = wAtt - 44;
    const headH = 92, top0 = 88, gapV = 8;
    const rail = L + HW + 12;                 // 右侧汇集线：把 4 个头的 out 收拢到 Σ 框
    const sumY = 492, sumH = 44;
    const taps = [];
    for (let i = 0; i < h; i++) {
      const yy = top0 + i * (headH + gapV);
      s += `<rect x="${L}" y="${yy}" width="${HW}" height="${headH}" rx="8" fill="#2a3040" stroke="#5b6172"/>`;
      s += `<text x="${L + 14}" y="${yy + 18}" fill="#9aa3b2" font-size="10">head ${i}</text>`;
      const qx = L + 40, kx = L + 98, vx = L + 156, midY = yy + 44;
      s += `<circle cx="${qx}" cy="${midY}" r="10" fill="#3f4763" stroke="#8a90a3"/>
            <circle cx="${kx}" cy="${midY}" r="10" fill="#3f4763" stroke="#8a90a3"/>
            <circle cx="${vx}" cy="${midY}" r="10" fill="#3f4763" stroke="#8a90a3"/>`;
      s += `<text x="${qx}" y="${midY + 5}" fill="#dfe3ea" font-size="11" text-anchor="middle">Q</text>
            <text x="${kx}" y="${midY + 5}" fill="#dfe3ea" font-size="11" text-anchor="middle">K</text>
            <text x="${vx}" y="${midY + 5}" fill="#dfe3ea" font-size="11" text-anchor="middle">V</text>`;
      const mx = L + HW / 2, oy = midY + 24;
      s += `<line x1="${kx}" y1="${midY}" x2="${mx}" y2="${oy}" stroke="#b18be0" stroke-width="1.4"/>
            <line x1="${vx}" y1="${midY}" x2="${mx + 20}" y2="${oy}" stroke="#b18be0" stroke-width="1.4"/>`;
      s += `<circle cx="${mx}" cy="${oy}" r="10" fill="none" stroke="#e0c25a" stroke-width="1.6"/>`;
      s += `<text x="${mx}" y="${oy + 5}" fill="#e0c25a" font-size="12" text-anchor="middle">⊗</text>`;
      s += `<text x="${L + HW - 8}" y="${yy + 18}" fill="#9aa3b2" font-size="9" text-anchor="end">softmax(QKᵀ/√d)</text>`;
      s += `<rect x="${L + HW - 40}" y="${oy + 6}" width="34" height="15" rx="4" fill="#3f4763"/>
            <text x="${L + HW - 23}" y="${oy + 17}" fill="#dfe3ea" font-size="9" text-anchor="middle">out</text>`;
      taps.push(oy + 13);                     // 该头 out 框的纵向中心
    }
    // 汇集线：每个头的 out → 右侧竖线 → 底部 Σ 框
    for (const ty of taps) {
      s += `<line x1="${L + HW - 6}" y1="${ty}" x2="${rail}" y2="${ty}" stroke="#c3a1ee" stroke-width="1.4"/>`;
    }
    s += `<line x1="${rail}" y1="${taps[0]}" x2="${rail}" y2="${sumY + sumH / 2}" stroke="#c3a1ee" stroke-width="1.4"/>`;
    s += `<line x1="${rail}" y1="${sumY + sumH / 2}" x2="${L + HW}" y2="${sumY + sumH / 2}" stroke="#c3a1ee" stroke-width="1.6" marker-end="url(#arrL)"/>`;
    // Σ 子层：频率视图（🎯 ATTENTION 频率）分析的就是这一层
    s += `<g data-part="attnSum">`;
    s += `<rect x="${L}" y="${sumY}" width="${HW}" height="${sumH}" rx="8" fill="#2a3040" stroke="#8a90a3"/>`;
    s += `<text x="${L + 14}" y="${sumY + 19}" fill="#e0c25a" font-size="13">Σ</text>`;
    s += `<text class="partName" x="${L + 32}" y="${sumY + 19}" fill="#dfe3ea" font-size="11">4 头输出相加 → W_O 投影</text>`;
    s += `<text x="${L + 14}" y="${sumY + 35}" fill="#9aa3b2" font-size="9.5">残差流增量（位置「=」）· ${d} 维</text>`;
    s += `</g>`;
    s += lbl(attCx, 580, "每个头：softmax(QKᵀ/√d_head)·V（d_head=" + (cfg ? cfg.d_head : 32) + "）；4 头输出相加后由 W_O 投影回 " + d + " 维", 10.5);
    s += `</g>`;          // 关闭 ATTENTION 分组
    s += `</g>`;          // 关闭 translate（MLP 用绝对坐标、占满全高，下面单独画）

    // 4) MLP —— 圆圈=结点、线=权重；输入/隐藏/输出均按“单列(1D)”排布，结点数真实（128 / 512 / 128）
    //    boxY=70 / boxH=480：与上方 ATTENTION 完全等高、同顶（70→550）
    {
      const boxY = 70, boxH = 480;
      s += `<g data-stage="mlp"><rect x="${mlpX}" y="${boxY}" width="${wMlp}" height="${boxH}" rx="12" fill="url(#bx)" stroke="#2e8b66"/>`;
      s += lbl(mlpCx, boxY - 15, "MLP · 单隐层全连接", 13, "#57c08a");   // 标题在框上方，与 ATTENTION 标题同线
      s += lbl(mlpCx, boxY + 18, `${d} → ${m} → ${d}（+ReLU）`, 10.5);

      // 三层结点各为一整列(1D)：x 固定，y 在 [yTop, yBot] 上等距（480 高度内 512 偏密，呈连续竖条）
      const xIn = mlpX + 52, xHid = mlpX + 165, xOut = mlpX + 278;
      const yTop = boxY + 74, yBot = boxY + boxH - 64;
      const col = (n) => {
        const st = (yBot - yTop) / (n - 1);
        return Array.from({ length: n }, (_, i) => yTop + i * st);
      };
      const yIn = col(d), yHid = col(m), yOut = col(d);
      const dotCol = (x, ys, r, fill, stroke) =>
        ys.map((y) => `<circle cx="${x}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="0.4"/>`).join("");

      // 权重=线：相邻单列之间按比例采样连线（示意“全连接”），先画线让结点盖在上面
      const fan = (xa, ya, xb, yb, color) => {
        const a0 = ya[0], a1 = ya[ya.length - 1], b0 = yb[0], b1 = yb[yb.length - 1];
        let t = "";
        for (let i = 0; i < 30; i++) {
          const tt = i / 29;
          t += `<line x1="${xa}" y1="${(a0 + tt * (a1 - a0)).toFixed(1)}"
                       x2="${xb}" y2="${(b0 + tt * (b1 - b0)).toFixed(1)}"
                       stroke="${color}" stroke-width="0.6" stroke-opacity="0.22"/>`;
        }
        return t;
      };
      // 两个权重层各画一个虚框（mlpIn / mlpOutW）：只是标出层次与尺寸，不作为点亮目标
      // （「MLP 频率」画的是 MLP 的输出激活，不是权重，见下面的 mlpOut）
      // 尺寸按 PyTorch 的存法标注：nn.Linear(in, out).weight.shape = [out, in]（行=输出维）
      const fTop = yTop - 40, fBot = yBot + 8;
      for (const fr of [
        { part: "mlpIn",   x1: xIn + 6,  x2: xHid - 6, title: `W_in ${m}×${d}`,  dir: `in ${d} → out ${m}`,  col: "#8fd0ff" },
        { part: "mlpOutW", x1: xHid + 6, x2: xOut - 6, title: `W_out ${d}×${m}`, dir: `in ${m} → out ${d}`, col: "#a7e0b0" },
      ]) {
        const cx = (fr.x1 + fr.x2) / 2;
        s += `<g data-part="${fr.part}">`;
        s += `<rect x="${fr.x1}" y="${fTop}" width="${fr.x2 - fr.x1}" height="${fBot - fTop}" rx="8" fill="none" stroke="#3d4a63" stroke-dasharray="5 5"/>`;
        s += `<text class="partName" x="${cx}" y="${fTop + 17}" fill="${fr.col}" font-size="10" text-anchor="middle" font-family="sans-serif">${fr.title}</text>`;
        s += `<text x="${cx}" y="${fTop + 30}" fill="#7f8ba3" font-size="8.5" text-anchor="middle" font-family="sans-serif">${fr.dir}</text>`;
        s += `</g>`;
      }

      s += `<g stroke-linecap="round">${fan(xIn + 10, yIn, xHid - 10, yHid, "#8fd0ff")}${fan(xHid + 10, yHid, xOut - 10, yOut, "#a7e0b0")}</g>`;

      s += dotCol(xIn, yIn, 1.3, "#3f4763", "#9db4d0");
      s += dotCol(xHid, yHid, 0.6, "#1e7d46", "#8ae3a8");
      // MLP 的输出向量（= W_out·ReLU(…) + b_out，加回残差流的 128 维增量）单独成组：
      // 「MLP 频率」画的就是这一列里的某一维，所以点亮的是它、而不是权重框
      s += `<g data-part="mlpOut">`;
      s += `<rect x="${xOut - 8}" y="${yTop - 10}" width="16" height="${(yBot - yTop + 20).toFixed(1)}" rx="8" fill="none" stroke="#3d4a63" stroke-dasharray="5 5"/>`;
      s += `<g class="partHl">${dotCol(xOut, yOut, 1.3, "#4a4f78", "#c9a15c")}</g>`;
      s += `</g>`;
      s += lbl(xIn, yBot + 22, `${d} 输入`, 10.5, "#dfe3ea");
      s += lbl(xHid, yBot + 22, `${m} 神经元 · ReLU`, 10.5, "#dfe3ea");
      s += lbl(xOut, yBot + 22, `${d} 输出`, 10.5, "#dfe3ea");
      s += lbl(mlpCx, boxY + boxH - 14, "（尺寸按 PyTorch 存法 = [输出维 × 输入维]；512 偏密，连线为部分示意）", 8.5);
      s += `</g>`;
    }
    // 重新平移：画 UNEMBED 等
    s += `<g transform="translate(0,${VS})">`;
    // 5) 读出端：把 UNEMBED 的权重矩阵 W_U 与它读出的 logits 向量并排画，行高对齐
    //    第 r 行 W_U[r] 与第 r 格 logit 是同一个输出类 —— 这样两个面板的区别一眼可见
    const RD_TOP = 150, RD_H = 312, RD_ROW = RD_H / p;
    const rdClass = (hasS && sample.c != null) ? sample.c : (aVal + bVal) % p;   // 真值类（金 = 真值）
    // 5a) UNEMBED：W_U = [113 类 × 128 维]，一行 = 一个类的读出向量（**权重**，静态）
    //     （权重视图已下线：UNEMBED 的“输出”就是 logits，由 📊 LOGITS 频率面板负责）
    s += `<g data-stage="unembed"><rect x="${unX}" y="120" width="${wUn}" height="380" rx="8" fill="url(#bx)" stroke="#c0843c"/>`;
    s += lbl(unCx, 100, "UNEMBED", 11, "#c9a15c");
    s += lbl(unCx, 138, `W_U ${p} 类 × ${d} 维`, 9, "#9aa3b2");
    {
      const gx = unX + 8, gw = wUn - 50;
      for (let r = 0; r < p; r++) {                 // 一行 = 一个输出类的读出向量
        s += `<rect x="${gx}" y="${(RD_TOP + r * RD_ROW).toFixed(1)}" width="${gw}" height="${Math.max(1, RD_ROW - 0.6).toFixed(2)}" fill="#273149"/>`;
      }
      for (let i = 1; i < 4; i++) {                 // 细竖线：示意 128 个残差维（此处压成几列）
        const vx = (gx + (gw * i) / 4).toFixed(1);
        s += `<line x1="${vx}" y1="${RD_TOP}" x2="${vx}" y2="${RD_TOP + RD_H}" stroke="#3a4257" stroke-width="0.6"/>`;
      }
      const hatY = RD_TOP + rdClass * RD_ROW;       // 金 = 真值类：它在 W_U 里是哪一行
      s += `<rect x="${gx - 3}" y="${(hatY - 1).toFixed(1)}" width="${gw + 6}" height="${(RD_ROW + 2).toFixed(1)}" rx="1.5" fill="#ffd57a" stroke="#ffd57a" stroke-width="0.6"/>`;
      s += `<text x="${gx + gw + 6}" y="${(hatY + 3.4).toFixed(1)}" fill="#ffd57a" font-size="9" font-family="monospace">类 ${rdClass}</text>`;
      s += lbl(unCx, 484, "每行 = 一个输出类的读出向量", 8, "#9aa3b2");
    }
    s += `</g>`;       // 关闭 UNEMBED 分组
    // 5b) logits：UNEMBED 读出的结果向量本身 = 113 个类的 logit（**激活**，取决于输入 a、b）
    //     「LOGITS 频率」画的是所选输出类那一格随 b 的变化，所以点亮的是那一格
    s += `<g data-stage="logits"><rect x="${logX}" y="120" width="${wLog}" height="380" rx="8" fill="url(#bx)" stroke="#c0843c"/>`;
    s += lbl(logCx, 100, "logits", 11, "#c9a15c");
    s += lbl(logCx, 138, `${p} 类`, 9, "#9aa3b2");
    {
      const gx = logX + 10, gw = wLog - 40;
      for (let r = 0; r < p; r++) {                 // 一格 = 一个输出类的 logit（行高与 W_U 对齐）
        const cy = (RD_TOP + r * RD_ROW).toFixed(1);
        s += `<g data-part="logitsCell" data-idx="${r}">`;
        s += `<rect x="${gx}" y="${cy}" width="${gw}" height="${Math.max(1, RD_ROW - 0.6).toFixed(2)}" fill="#273149"/>`;
        s += `<text x="${gx + gw + 5}" y="${(RD_TOP + r * RD_ROW + RD_ROW - 0.4).toFixed(1)}" fill="#8fe3f2" font-size="8" font-family="monospace">${r}</text>`;
        s += `</g>`;
      }
      const tY = RD_TOP + rdClass * RD_ROW;         // 同一个真值类，在这边是哪一格
      s += `<rect x="${gx - 3}" y="${(tY - 1).toFixed(1)}" width="${gw + 6}" height="${(RD_ROW + 2).toFixed(1)}" rx="1.5" fill="none" stroke="#ffd57a" stroke-width="0.8"/>`;
      s += lbl(logCx, 484, "每格 = 一类的 logit", 7.5, "#9aa3b2");
    }
    s += `</g>`;       // 关闭 logits 分组
    s += `</g>`;       // 关闭 translate 分组

    // 测试后的预测结果显示在输出端
    if (hasS && sample.pred != null) {
      const ok = !!sample.correct;
      s += `<text x="${logCx}" y="566" fill="${ok ? '#6ee7a0' : '#ff8a80'}" font-size="12.5" text-anchor="middle" font-family="sans-serif">ĉ = ${sample.pred}${ok ? " ✓" : " ✗"}</text>`;
    }

    svg.innerHTML = s;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    highlightArch(state.archStage || null);   // 重画（如测试会重建结构图）后恢复高亮
  }

  // ===================================================================
  // 训练曲线 + 选点
  // ===================================================================
  function grokEpochOf(curve) {
    if (!curve || !curve.test_acc) return null;
    const ep = curve.epoch, acc = curve.test_acc;
    for (let i = 0; i < ep.length; i++) if (acc[i] >= 0.95) return ep[i];
    return null;
  }
  function stageOf(epoch) {
    const g = grokEpochOf(state.idx?.curve);
    if (g == null) return "";
    if (epoch < g * 0.55) return "记忆期 (memorization)";
    if (epoch < g) return "电路形成期 (circuit formation)";
    if (epoch < g * 1.5) return "清理期 (cleanup)";
    return "收敛 (泛化解)";
  }

  function drawCurve() {
    const cv = $("#cCurve");
    const W = 1100, H = 300;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const curve = state.idx?.curve;
    if (!curve) return;
    const ep = curve.epoch;
    const e0 = ep[0], e1 = ep[ep.length - 1];
    const X = (e) => 60 + ((e - e0) / (e1 - e0)) * (W - 90);
    const drawLoss = (key, color) => {
      const L = curve[key];
      if (!L) return;
      const vals = L.map((v) => Math.max(v, 1e-9));
      const lmin = Math.log10(Math.min(...vals)), lmax = Math.log10(Math.max(...vals));
      const Y = (v) => H - 30 - ((Math.log10(Math.max(v, 1e-9)) - lmin) / (lmax - lmin)) * (H - 70);
      ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath();
      ep.forEach((e, i) => { const px = X(e), py = Y(L[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
    };
    drawLoss("test_loss", "#d64541");
    drawLoss("train_loss", "#2c6fbb");

    // 当前 epoch 标记
    if (state.loadedEpoch != null) {
      const px = X(state.loadedEpoch);
      ctx.strokeStyle = "#8b2f2f"; ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(px, 18); ctx.lineTo(px, H - 26); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#8b2f2f"; ctx.font = "12px sans-serif";
      ctx.fillText(state.loadedEpoch, px - 16, H - 8);
    }
    // 轴
    ctx.fillStyle = "#57606a"; ctx.font = "11px sans-serif";
    ctx.fillText("epoch →", W - 70, H - 8);
    ctx.fillText("loss (log) ↑", 8, 16);

    // 交互：点击/拖动设 epoch（找最近可用快照）
    const setFromEvent = (ev) => {
      const rect = cv.getBoundingClientRect();
      const frac = (ev.clientX - rect.left) / rect.width;
      const epTarget = e0 + frac * (e1 - e0);
      const nearest = state.epochs.reduce((a, b) =>
        Math.abs(b - epTarget) < Math.abs(a - epTarget) ? b : a, state.epochs[0]);
      $("#epSlider").value = state.epochs.indexOf(nearest);
      selectEpoch(nearest);
    };
    cv.onpointerdown = (ev) => { cv.setPointerCapture(ev.pointerId); setFromEvent(ev); };
    cv.onpointermove = (ev) => { if (ev.buttons & 1) setFromEvent(ev); };
  }

  // ===================================================================
  // 视图渲染
  // ===================================================================
  function matView() {
    if (!state.mats) return;
    const mat = (n) => state.mats[n];
    const get = (m) => (i) => m.data[i];

    // Embedding W_E 全表 + 前 p 行 Fourier
    const emb = mat("token_embed");
    renderHeat($("#cWE"), emb.shape, get(emb), { transpose: true });
    const p = state.cfg.p;
    const colsE = emb.shape[1];
    const sub = new Float32Array(p * colsE);
    for (let r = 0; r < p; r++) for (let c = 0; c < colsE; c++) sub[r * colsE + c] = emb.data[r * colsE + c];
    const nE = fourierNorms1d(sub, p, colsE);
    renderBars($("#cWEf"), nE, Array.from({ length: nE.length }, (_, i) => i));

    // W_L = W_U @ W_out
    const U = mat("unembed");          // [p, d]
    const O = mat("mlp_out.w");        // [d, n_mlp]
    const d = state.cfg.d_model, n = state.cfg.n_mlp;
    const WL = new Float32Array(p * n);
    for (let a = 0; a < p; a++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < d; k++) s += U.data[a * d + k] * O.data[k * n + j];
        WL[a * n + j] = s;
      }
    }
    renderHeat($("#cWL"), [p, n], (i) => WL[i]);
    const nL = fourierNorms1d(WL, p, n);
    renderBars($("#cWLf"), nL, Array.from({ length: nL.length }, (_, i) => i));

    // 注意力头
    const head = Number($("#headSel").value);
    const which = $("#matSel").value;
    const Hm = mat(`head${head}.${which}`);
    renderHeat($("#cHead"), Hm.shape, get(Hm), { transpose: true });

    // MLP
    const Wi = mat("mlp_in.w");
    renderHeat($("#cWin"), Wi.shape, get(Wi), { transpose: true });
    const Wo = mat("mlp_out.w");
    renderHeat($("#cWout"), Wo.shape, get(Wo), { transpose: true });
  }

  // ===================================================================
  // 载入
  // ===================================================================
  async function selectEpoch(epoch) {
    state.gen = (state.gen || 0) + 1;
    const gen = state.gen;
    state.loadedEpoch = epoch;
    $("#epLabel").textContent = `${epoch} / ${state.epochs[state.epochs.length - 1]}`;
    const pill = $("#stagePill");
    const st = stageOf(epoch);
    pill.style.display = st ? "inline-block" : "none";
    pill.textContent = st;
    drawCurve();
    if (state.epochs.indexOf(epoch) !== Number($("#epSlider").value)) {
      $("#epSlider").value = state.epochs.indexOf(epoch);
    }
    await loadWeights(epoch);
    if (gen !== state.gen) return;          // 期间又选择了更新的阶段，丢弃这次刷新
    // 移动阶段不隐藏面板：对仍在显示的测试 / MLP 循环结果，用新权重自动重算
    await refreshPanels();
  }

  /** 按新权重刷新仍在显示的临时结果（测试 / 频率视图，二选一） */
  async function refreshPanels() {
    if (!state.mats) return;
    const tr = $("#testResult"), ed = $("#freqResult");
    if (ed && !ed.hidden && state.freqSrc) { await runFreqView(state.freqSrc); return; }
    if (tr && !tr.hidden && state.lastTest) { await runTest(state.lastTest); }
  }

  async function loadWeights(epoch) {
    const btn = $("#loadBtn");
    btn.disabled = true; btn.innerHTML = '<span class="spin"></span> 载入中…';
    try {
      const file = `e${pad6(epoch)}.bin`;
      const base = `/viz-data/${state.run}/seed${state.seed}/${file}`;
      const resp = await fetch(base);
      if (!resp.ok) throw new Error(`${file} ${resp.status}`);
      const buf = await resp.arrayBuffer();
      const { mats } = parseBin(buf);
      state.mats = mats;
      state.freqPick = {};          // 自动挑中的维度作废，下次打开重新挑
      matView();
    } catch (err) {
      console.error(err);
      alert(`载入权重失败：${err.message}`);
    } finally {
      btn.disabled = false; btn.innerHTML = "载入权重";
    }
  }

  async function onSeed() {
    state.seed = $("#seedSel").value;
    const idx = state.index.runs[state.run].seeds[state.seed];
    state.idx = idx;
    state.cfg = idx.config;
    state.epochs = (idx.epochs || []).slice().sort((a, b) => a - b);
    clearTest();
    $("#dims").textContent = `P=${idx.config.p} · d=${idx.config.d_model} · heads=${idx.config.n_heads} · MLP=${idx.config.n_mlp}`;
    buildArch(idx.config);
    if (state.epochs.length === 0) {
      $("#epLabel").textContent = "（无可用快照）";
      return;
    }

    // 滑块 = 快照下标
    const epSlider = $("#epSlider");
    epSlider.max = state.epochs.length - 1;
    epSlider.value = state.epochs.length - 1;   // 默认载入最终
    epSlider.oninput = () => selectEpoch(state.epochs[Number(epSlider.value)]);
    drawCurve();
    // head 下拉
    const hs = $("#headSel");
    hs.innerHTML = "";
    for (let i = 0; i < idx.config.n_heads; i++) hs.add(new Option(`head ${i}`, String(i)));
    selectEpoch(state.epochs[state.epochs.length - 1]);
  }

  // ===================================================================
  // 测试：取一个测试样本 → 前向传播 → logits
  // ===================================================================
  /** 按钮下方那行说明：随当前打开的面板变化（key 与各面板一一对应；
   *  freq:* 的文案在 setPanelInfo 里由 FREQ_SRC 现场拼出来，避免两处重复维护） */
  const PANEL_INFO = {
    none: "点上面的按钮运行，结果出现在下方面板（都用<b>当前阶段已加载的权重</b>计算）。" +
          "4 个按钮沿数据流逐级看<b>频率</b>：EMBED → ATTENTION → MLP → LOGITS；" +
          "切换按钮即切换视图，拖动 epoch 会用新权重自动重算。",
    test: "<b>测试</b>：从该 seed 的测试集里随机取一组 (a, b) 跑一次前向，得到 113 个输出类的 logits，" +
          "并把预测 ĉ 在结构图上点亮（金色下划线 = 真值）。",
  };

  function setPanelInfo(key) {
    const el = $("#testInfo");
    let txt = PANEL_INFO[key];
    let btnId = key === "test" ? "testBtn" : null;
    let stage = null;
    if (!txt && key && key.startsWith("freq:")) {          // 频率视图：从 FREQ_SRC 取短文
      const sp = FREQ_SRC[key.slice(5)];
      if (sp) { txt = `<b>${sp.name}</b>：${sp.hint}`; btnId = sp.btn; stage = sp.stage; }
    }
    // 选中的按钮高亮（背景变亮）；无面板打开时全部取消
    for (const b of document.querySelectorAll(".test-bar button")) b.classList.toggle("active", b.id === btnId);
    highlightArch(stage);                                   // 结构图上同步高亮对应的一级
    if (el) el.innerHTML = txt || PANEL_INFO.none;
  }

  /** stage → 该级里被“细化点亮”的子元素（由 FREQ_SRC[*].vec / .part 声明；运行时查，避免初始化顺序问题）
   *  vec  = EMBED 里某条 token 嵌入向量（W_E[a]）
   *  part = 层内某个具体子层（如 ATTENTION 里的「Σ 4 头输出 → W_O」） */
  function stageFine(stage) {
    if (!stage) return null;
    for (const k of Object.keys(FREQ_SRC)) {
      const sp = FREQ_SRC[k];
      if (sp.stage === stage && (sp.vec || sp.part)) return { vec: sp.vec || null, part: sp.part || null };
    }
    return null;
  }
  function highlightArch(stage) {
    state.archStage = stage || null;
    const fine = stageFine(stage);
    // LOGITS 是一整排格子（一格 = 一个输出类）：要跟着面板里选的输出类移动
    let pickIdx = null;
    if (fine && fine.part === "logitsCell") {
      const sel = $("#freqSel");
      if (sel && sel.value !== "") pickIdx = Number(sel.value);
    }
    for (const g of document.querySelectorAll("#archSvg g[data-stage]")) {
      // 有细化子元素时不再整块高亮（EMBED → W_E[a]；ATTENTION → Σ 头输出；……）
      g.classList.toggle("on", g.dataset.stage === stage && !fine);
    }
    for (const g of document.querySelectorAll("#archSvg g[data-vec]")) {
      g.classList.toggle("on", !!fine && g.dataset.vec === fine.vec);
    }
    for (const g of document.querySelectorAll("#archSvg g[data-part]")) {
      let on = !!fine && g.dataset.part === fine.part;
      if (on && g.dataset.idx != null && pickIdx != null) on = Number(g.dataset.idx) === pickIdx;
      g.classList.toggle("on", on);
    }
  }

  function clearTest() {
    state.sample = null;
    state.testList = null;
    state.testKey = null;
    state.lastTest = null;
    const r = $("#testResult");
    if (r) r.hidden = true;
    const ed = $("#freqResult");
    if (ed) ed.hidden = true;
    state.freqSrc = null;
    setPanelInfo("none");
  }

  /** 无 test_pairs.json 时的兜底：按 1-frac 概率接受，近似“留出”样本（可复现、非 numpy 精确划分） */
  function pseudoTestSample(p, frac, seed) {
    let a = (seed ^ 0x9e3779b9) >>> 0;
    const rnd = () => {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    for (let g = 0; g < 20000; g++) {
      const aa = Math.floor(rnd() * p);
      const bb = Math.floor(rnd() * p);
      if (rnd() >= frac) return { a: aa, b: bb, c: (aa + bb) % p };
    }
    return { a: 0, b: 1, c: 1 };
  }

  async function ensureTestList() {
    const run = state.run, seed = state.seed;
    const key = `${run}#${seed}`;
    if (state.testKey === key && state.testList) return state.testList;
    state.testKey = key;
    try {
      const resp = await fetch(`/viz-data/${run}/seed${seed}/test_pairs.json`);
      if (!resp.ok) throw new Error(String(resp.status));
      state.testList = await resp.json();
    } catch (e) {
      state.testList = null;               // 回退到 JS 伪留出
    }
    return state.testList;
  }

  /** 复用的暂存缓冲：全分辨率网格要跑上万次前向，避免每次新建 ~20 个 TypedArray */
  let _ws = null;
  function workspace(d, dh, nh, nm, p) {
    const key = [d, dh, nh, nm, p].join("|");
    if (_ws && _ws.key === key) return _ws;
    _ws = {
      key,
      x: new Float64Array(3 * d), x0b: new Float64Array(3 * d), acc: new Float64Array(3 * d),
      q: new Float64Array(3 * dh), k: new Float64Array(3 * dh), v: new Float64Array(3 * dh),
      attn: new Float64Array(9), srow: new Float64Array(3), ctx: new Float64Array(3 * dh),
      resid2: new Float64Array(d), o: new Float64Array(d), x2: new Float64Array(d),
      h: new Float64Array(nm), logits: new Float64Array(p),
    };
    return _ws;
  }

  /** 单样本前向传播（与 python/grokking/model.py 的 forward_full 一致）。
   *  结果写进复用的暂存缓冲（ws.logits / ws.h），请立即读取；需要长期持有请用 forwardOne()。 */
  function forwardCore(a, b) {
    const { cfg, mats } = state;
    const p = cfg.p, d = cfg.d_model, dh = cfg.d_head, nh = cfg.n_heads, nm = cfg.n_mlp;
    const ws = workspace(d, dh, nh, nm, p);
    const M = (n) => { const m = mats[n]; return m ? m.data : null; };
    const te = M("token_embed"), pe = M("pos_embed");   // [(p+1),d] / [3,d]
    const tok = [a, b, p];                                // '=' 的 token id = p
    const x = ws.x, x0b = ws.x0b, attnAcc = ws.acc;        // x0 = W_E tok + pos
    for (let t = 0; t < 3; t++) {
      const ot = tok[t] * d;
      for (let c = 0; c < d; c++) x[t * d + c] = te[ot + c] + pe[t * d + c];
    }
    // 注意：torch 中所有 head 都基于同一个 x(0) 计算，头输出累加后再一次性残差相加。
    for (let i = 0; i < 3 * d; i++) x0b[i] = x[i];
    attnAcc.fill(0, 0, 3 * d);
    const invS = 1 / Math.sqrt(dh);
    for (let hi = 0; hi < nh; hi++) {
      const WQ = M(`head${hi}.W_Q`), WK = M(`head${hi}.W_K`),
            WV = M(`head${hi}.W_V`), WO = M(`head${hi}.W_O`);
      const q = ws.q, k = ws.k, v = ws.v;
      // W_Q/W_K/W_V 存为 [d_head, d]（out×in）；W_O 存为 [d, d_head]
      for (let t = 0; t < 3; t++) for (let j = 0; j < dh; j++) {
        let sq = 0, sk = 0, sv = 0;
        for (let c = 0; c < d; c++) {
          const xv = x0b[t * d + c];
          sq += xv * WQ[j * d + c]; sk += xv * WK[j * d + c]; sv += xv * WV[j * d + c];
        }
        q[t * dh + j] = sq; k[t * dh + j] = sk; v[t * dh + j] = sv;
      }
      // attn[t1][t2] = softmax_t2( q_t1 · k_t2 / sqrt(d_head) )
      const attn = ws.attn, srow = ws.srow;
      for (let t = 0; t < 3; t++) {
        let mx = -Infinity;
        for (let t2 = 0; t2 < 3; t2++) {
          let s = 0;
          for (let j = 0; j < dh; j++) s += q[t * dh + j] * k[t2 * dh + j];
          s *= invS; srow[t2] = s; if (s > mx) mx = s;
        }
        let es = 0;
        for (let t2 = 0; t2 < 3; t2++) { const e = Math.exp(srow[t2] - mx); attn[t * 3 + t2] = e; es += e; }
        for (let t2 = 0; t2 < 3; t2++) attn[t * 3 + t2] /= es;
      }
      const ctx = ws.ctx;
      for (let t = 0; t < 3; t++) for (let j = 0; j < dh; j++) {
        let s = 0;
        for (let t2 = 0; t2 < 3; t2++) s += attn[t * 3 + t2] * v[t2 * dh + j];
        ctx[t * dh + j] = s;
      }
      for (let t = 0; t < 3; t++) for (let c = 0; c < d; c++) {
        let s = 0;
        for (let j = 0; j < dh; j++) s += ctx[t * dh + j] * WO[c * dh + j];
        attnAcc[t * d + c] += s;                          // 先累加所有头的输出
      }
    }
    for (let i = 0; i < 3 * d; i++) x[i] = x0b[i] + attnAcc[i];   // 再一次性残差
    const resid2 = ws.resid2;
    for (let c = 0; c < d; c++) resid2[c] = x[2 * d + c];  // 取 '=' 位置的残差流
    const Wi = M("mlp_in.w"), bi = M("mlp_in.b");
    const h = ws.h;
    for (let j = 0; j < nm; j++) {
      let s = bi ? bi[j] : 0;
      for (let c = 0; c < d; c++) s += Wi[j * d + c] * resid2[c];
      h[j] = s > 0 ? s : 0;                               // ReLU
    }
    const Wo = M("mlp_out.w"), bo = M("mlp_out.b");
    const o = ws.o;
    for (let c = 0; c < d; c++) {
      let s = bo ? bo[c] : 0;
      for (let j = 0; j < nm; j++) s += Wo[c * nm + j] * h[j];
      o[c] = s;
    }
    const x2 = ws.x2;
    for (let c = 0; c < d; c++) x2[c] = resid2[c] + o[c];  // MLP 残差（含 b_out）
    const U = M("unembed");                               // [p,d]
    const logits = ws.logits;
    for (let r = 0; r < p; r++) {
      let s = 0;
      for (let c = 0; c < d; c++) s += U[r * d + c] * x2[c];
      logits[r] = s;
    }
    return ws;
  }

  /** 对外接口：返回本次前向的副本（可安全持有）
   *  attn / mlp 都是位置「=」处加回残差流的增量向量（各 d 维），供“频率视图”逐级看谱。 */
  function forwardOne(a, b) {
    const ws = forwardCore(a, b);
    const d = state.cfg.d_model;
    return {
      logits: ws.logits.slice(),
      h: ws.h.slice(),
      attn: ws.acc.slice(2 * d, 3 * d),   // 4 个头在位置「=」的输出之和
      mlp: ws.o.slice(),                  // MLP 输出（W_out·ReLU(...) + b_out）
      resid: ws.x2.slice(),               // 最终残差流（位置「=」，含 pos/attn/mlp）
    };
  }

  /** 画 logits 柱状图（金色=真值，蓝箭头=argmax） */
  function drawTestLogits(canvas, logits, truth) {
    const W = 1100, H = 180;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    const p = logits.length;
    let pred = 0;
    for (let i = 1; i < p; i++) if (logits[i] > logits[pred]) pred = i;
    const mx = Math.max(...logits), mn = Math.min(...logits);
    const span = mx - mn || 1;
    const pad = 10, baseY = H - 26, topY = 14;
    const bw = (W - pad * 2) / p;
    for (let i = 0; i < p; i++) {
      const t = (logits[i] - mn) / span;
      const bh = Math.max(1, t * (baseY - topY));
      ctx.fillStyle = i === truth ? "#ffcf5c" : (i === pred ? "#5a9ef0" : "#3c4a63");
      ctx.fillRect(pad + i * bw, baseY - bh, Math.max(1, bw - 0.8), bh);
    }
    ctx.fillStyle = "#7f8aa0"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
    const tick = Math.max(1, Math.ceil(p / 14));
    for (let i = 0; i < p; i += tick) ctx.fillText(String(i), pad + i * bw + bw / 2, H - 8);
    const xt = pad + truth * bw + bw / 2, xp = pad + pred * bw + bw / 2;
    ctx.strokeStyle = "#ffcf5c"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xt, topY - 6); ctx.lineTo(xt, baseY + 4); ctx.stroke();
    ctx.fillStyle = "#ffcf5c"; ctx.textAlign = "left";
    ctx.fillText(`真值 ${truth}`, Math.min(xt, W - 60), topY - 4);
    ctx.strokeStyle = "#5a9ef0";
    ctx.beginPath();
    ctx.moveTo(xp, topY - 6); ctx.lineTo(xp - 7, topY - 14);
    ctx.moveTo(xp, topY - 6); ctx.lineTo(xp + 7, topY - 14); ctx.stroke();
    ctx.fillStyle = "#5a9ef0";
    ctx.fillText(`argmax ${pred}`, Math.min(xp + 10, W - 70), topY - 8);
  }

  async function runTest(force) {
    const btn = $("#testBtn");
    if (!state.cfg || !state.epochs || state.epochs.length === 0) return;
    // 互斥：点“测试”时隐藏频率视图面板
    const edX = $("#freqResult");
    if (edX) edX.hidden = true;
    setPanelInfo("test");
    if (!state.mats) {
      const ep = state.loadedEpoch != null ? state.loadedEpoch : state.epochs[state.epochs.length - 1];
      await loadWeights(ep);
    }
    if (!state.mats) { alert("请先载入权重（点上方“载入权重”）"); return; }
    btn.disabled = true; btn.innerHTML = "运行中…";
    try {
      const cfg = state.cfg;
      let a, b, c, src;
      if (force && Number.isInteger(force.a)) {     // 阶段切换后的自动刷新：沿用上一次样本
        a = force.a; b = force.b; c = force.c; src = force.src;
      } else {
        const list = await ensureTestList();
        if (list && list.a && list.a.length) {
          const i = Math.floor(Math.random() * list.a.length);
          a = list.a[i]; b = list.b[i]; c = list.c[i];
          src = `${list.kind || "测试集"} · 测试对共 ${list.n_test_total}`;
        } else {
          const s = pseudoTestSample(cfg.p, cfg.frac_train || 0.3, Number(state.seed));
          a = s.a; b = s.b; c = s.c;
          src = "JS 近似留出（缺 test_pairs.json）";
        }
      }
      const { logits } = forwardOne(a, b);
      let pred = 0;
      for (let i = 1; i < logits.length; i++) if (logits[i] > logits[pred]) pred = i;
      const correct = pred === c;
      const order = Array.from({ length: logits.length }, (_, i) => i)
        .sort((x, y) => logits[y] - logits[x]).slice(0, 5);
      const maxL = logits[order[0]];
      const exps = order.map((i) => Math.exp(logits[i] - maxL));
      const esum = exps.reduce((s, e) => s + e, 0);

      const R = $("#testResult"); R.hidden = false;
      $("#trIn").textContent = `a=${a}, b=${b}`;
      $("#trC").textContent = String(c);
      $("#trPred").innerHTML =
        `${pred} <span class="${correct ? 'badge-ok' : 'badge-no'}">${correct ? '正确 ✓' : '错误 ✗'}</span>`;
      $("#trSrc").textContent = src;
      const st = stageOf(state.loadedEpoch);
      $("#trStage").textContent = `${state.loadedEpoch} 步${st ? " · " + st : ""}`;
      $("#trTop").innerHTML = order.map((i, rk) =>
        `<li><span class="rk">${rk + 1}</span><span class="cl ${i === c ? 'badge-ok' : ''}">${i}${i === c ? ' ▸' : ''}</span>` +
        `<span class="lg">${logits[i].toFixed(3)}</span><span class="prob">${((exps[rk] / esum) * 100).toFixed(2)}%</span></li>`
      ).join("");
      drawTestLogits($("#cTestLogits"), logits, c);

      // 在结构图中点亮该样本 & 显示预测
      state.sample = { a, b, c, pred, correct };
      state.lastTest = { a, b, c, src };   // 记住样本，切阶段后可沿用重算
      buildArch(cfg, state.sample);
    } catch (err) {
      console.error(err);
      alert(`测试失败：${err.message}`);
    } finally {
      btn.disabled = false; btn.innerHTML = "🧪 测试（前向推理）";
    }
  }

  // ===================================================================
  // 频率视图的三张图（被五级视图共用）：① 波形 vs 重建 ② 频谱 ③ 拆解与叠加
  // ===================================================================
  /** ① 原始曲线（淡蓝）vs 只用前 N 个频率重建的曲线（橙） */
  function drawEmbedWave(canvas, p, y, yrec, info) {
    const W = 1100, H = 300;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const L = 66, R = 16, T = 28, B = 40;
    const pw = W - L - R, ph = H - T - B;
    let vmin = 0, vmax = 0;
    for (let n = 0; n < p; n++) {
      vmax = Math.max(vmax, y[n], yrec[n]);
      vmin = Math.min(vmin, y[n], yrec[n]);
    }
    if (vmax - vmin < 1e-12) vmax = vmin + 1;
    const X = (n) => L + (pw * n) / (p - 1);
    const Y = (v) => T + ph - ((ph * (v - vmin)) / (vmax - vmin));
    const fmt = (v) => (v === 0 ? "0" : Math.abs(v) < 0.01 ? v.toExponential(1) : v.toFixed(3));

    ctx.strokeStyle = "#eef0f4"; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const gy = T + (ph * g) / 4;
      ctx.beginPath(); ctx.moveTo(L, gy); ctx.lineTo(W - R, gy); ctx.stroke();
    }
    ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    for (let g = 0; g <= 4; g++) {
      const v = vmax - ((vmax - vmin) * g) / 4;
      ctx.fillText(fmt(v), L - 8, T + (ph * g) / 4 + 4);
    }
    if (vmin < 0 && vmax > 0) {                     // 0 基线
      ctx.strokeStyle = "#c9ced7"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(W - R, Y(0)); ctx.stroke();
      ctx.setLineDash([]);
    }
    const line = (arr, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.beginPath();
      for (let n = 0; n < p; n++) { const px = X(n), py = Y(arr[n]); n ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke();
    };
    line(y, "#9db4cc", 1.4);                        // 原始
    line(yrec, "#e07b1a", 2.2);                     // 重建

    // 标题 + 内联图例（图例接在标题后面，避免和 x 轴刻度抢位置）
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    const title = `${info.qty}（第 ${info.c} 项）· ${info.xAxis} 0…${p - 1}`;
    ctx.fillText(title, L, 17);
    let lx = L + ctx.measureText(title).width + 18;
    const legLabel = `原始 ${info.qty}`;
    ctx.strokeStyle = "#9db4cc"; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(lx, 13); ctx.lineTo(lx + 24, 13); ctx.stroke();
    ctx.fillStyle = "#57606a"; ctx.font = "11px sans-serif";
    ctx.fillText(legLabel, lx + 29, 17);
    lx += 29 + ctx.measureText(legLabel).width + 16;
    ctx.strokeStyle = "#e07b1a"; ctx.lineWidth = 2.2;
    ctx.beginPath(); ctx.moveTo(lx, 13); ctx.lineTo(lx + 24, 13); ctx.stroke();
    ctx.fillStyle = "#57606a";
    ctx.fillText("只用前 N 个频率重建", lx + 29, 17);

    ctx.textAlign = "right"; ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
    ctx.fillText(`保留 k = ${info.topN.map((t) => t.k).join(", ")} · 残差 RMS ${fmt(info.rms)}（占能量 ${info.errPct.toFixed(2)}%）`,
      W - R, 17);

    ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
    const tick = Math.max(1, Math.ceil(p / 16));
    for (let n = 0; n < p; n += tick) {
      const isLast = n + tick >= p;                 // 末刻度右对齐，避免和右边界/轴名相撞
      ctx.textAlign = isLast ? "right" : "center";
      ctx.fillText(String(n), isLast ? W - R : X(n), T + ph + 18);
    }
    ctx.textAlign = "right"; ctx.fillStyle = "#333"; ctx.font = "12px sans-serif";
    ctx.fillText(`${info.xAxis} →`, W - R, H - 5);
  }

  /** ② 频率谱：柱高 = |c_k|，红 = 被保留的频率 */
  function drawEmbedSpec(canvas, p, series, keep) {
    const W = 1100, H = 220;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const L = 66, R = 16, T = 26, B = 38;
    const pw = W - L - R, ph = H - T - B;
    const comps = series.comps, K = comps.length;
    let maxA = 1e-12;
    for (const c of comps) maxA = Math.max(maxA, c.amp);
    const bw = pw / K;
    const X = (i) => L + i * bw;

    ctx.strokeStyle = "#eef0f4"; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const gy = T + (ph * g) / 4;
      ctx.beginPath(); ctx.moveTo(L, gy); ctx.lineTo(W - R, gy); ctx.stroke();
    }
    for (let i = 0; i < K; i++) {
      const c = comps[i];
      const h = (ph * c.amp) / maxA;
      const on = keep.has(c.k);
      ctx.fillStyle = on ? "#d64541" : "#b9c8dc";
      ctx.fillRect(X(i) + 1, T + ph - h, Math.max(1.5, bw - 2), h);
      if (on) {
        ctx.fillStyle = "#b0332f"; ctx.font = "bold 11px sans-serif"; ctx.textAlign = "center";
        ctx.fillText(String(c.k), X(i) + bw / 2, T + ph - h - 5);
      }
    }
    ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    ctx.fillText(maxA.toExponential(2), L - 8, T + 9);
    ctx.fillText("0", L - 8, T + ph + 4);
    ctx.textAlign = "center";
    const tick = Math.max(1, Math.ceil(K / 16));
    for (let i = 0; i < K; i += tick) ctx.fillText(String(comps[i].k), X(i) + bw / 2, T + ph + 16);
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    ctx.fillText("|c_k|  每个频率的幅度（红 = 保留）", L, 15);
    ctx.textAlign = "right"; ctx.fillStyle = "#333"; ctx.font = "12px sans-serif";
    ctx.fillText("频率 k →", W - R, H - 6);
  }

  /** ③ 拆解与叠加：上半 = 把保留的频率一条条加上去、逐步逼近原始；下半 = 每条波单独的样子 */
  function drawEmbedParts(canvas, p, y, series, topN, yrec, xAxis = "token a") {
    const n = topN.length;
    const nCols = Math.min(4, n);
    const nRowsB = Math.ceil(n / nCols);
    const W = 1100, topH = 330, cellH = 208;
    const H = topH + 12 + nRowsB * cellH + 28;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const X = (ox, w, i) => ox + (w * i) / (p - 1);
    const fmt = (v) => (v === 0 ? "0" : Math.abs(v) < 0.01 ? v.toExponential(1) : v.toFixed(3));

    // 前缀和：partials[j] = 直流 + 前 j 个频率（j = 0…n）
    const partials = [new Float64Array(p).fill(series.c0)];
    for (let j = 0; j < n; j++) {
      const t = topN[j], prev = partials[j], cur = new Float64Array(p);
      for (let i = 0; i < p; i++) {
        const th = (2 * Math.PI * t.k * i) / p;
        cur[i] = prev[i] + t.a * Math.cos(th) + t.b * Math.sin(th);
      }
      partials.push(cur);
    }

    // ---------- 上半：逐个频率叠加 ----------
    {
      const L = 66, R = 16, T = 26, B = 30, ow = W - L - R, oh = topH - T - B;
      let vmin = 0, vmax = 0;
      const scan = (arr) => { for (let i = 0; i < p; i++) { if (arr[i] > vmax) vmax = arr[i]; if (arr[i] < vmin) vmin = arr[i]; } };
      scan(y);
      for (const s of partials) scan(s);
      if (vmax - vmin < 1e-12) vmax = vmin + 1;
      const Y = (v) => T + oh - ((oh * (v - vmin)) / (vmax - vmin));

      ctx.strokeStyle = "#eef0f4"; ctx.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        const gy = T + (oh * g) / 4;
        ctx.beginPath(); ctx.moveTo(L, gy); ctx.lineTo(W - R, gy); ctx.stroke();
      }
      ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
      for (let g = 0; g <= 4; g++) {
        ctx.fillText(fmt(vmax - ((vmax - vmin) * g) / 4), L - 8, T + (oh * g) / 4 + 4);
      }
      if (vmin < 0 && vmax > 0) {
        ctx.strokeStyle = "#c9ced7"; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(W - R, Y(0)); ctx.stroke();
        ctx.setLineDash([]);
      }
      const line = (arr, color, width) => {
        ctx.strokeStyle = color; ctx.lineWidth = width;
        ctx.beginPath();
        for (let i = 0; i < p; i++) { const px = X(L, ow, i), py = Y(arr[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
        ctx.stroke();
      };
      line(y, "#8fa8c4", 2.4);                                   // 原始
      for (let j = 1; j < n; j++) {                              // 中间结果：浅 → 深
        const l = 80 - (24 * (j - 1)) / Math.max(1, n - 1);
        line(partials[j], `hsl(28, 80%, ${l}%)`, 1.6);
      }
      if (n >= 1) line(partials[n], "#e07b1a", 2.2);              // 加满 = 重建

      // 标题 + 内联图例
      ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
      const title = `逐个频率叠加：直流 c₀ = ${fmt(series.c0)}`;
      ctx.fillText(title, L, 17);
      let lx = L + ctx.measureText(title).width + 18;
      const swatch = (color, label, w = 24) => {
        ctx.strokeStyle = color; ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(lx, 13); ctx.lineTo(lx + w, 13); ctx.stroke();
        ctx.fillStyle = "#57606a"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
        ctx.fillText(label, lx + w + 5, 17);
        lx += w + 5 + ctx.measureText(label).width + 16;
      };
      swatch("#8fa8c4", "原始");
      for (let j = 1; j <= n; j++) {
        const l = j === n ? 50 : 80 - (24 * (j - 1)) / Math.max(1, n - 1);
        swatch(`hsl(28, 80%, ${l}%)`, j === n ? `+ 全部（= 重建）` : `+ k=${topN[j - 1].k}`, j === n ? 34 : 18);
      }
      ctx.textAlign = "right"; ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
      ctx.fillText(`每加一个频率就更贴近灰色；加满 ${n} 个后基本重合`, W - R, 17);

      ctx.textAlign = "center"; ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
      const tick = Math.max(1, Math.ceil(p / 16));
      for (let i = 0; i < p; i += tick) {
        const isLast = i + tick >= p;
        ctx.textAlign = isLast ? "right" : "center";
        ctx.fillText(String(i), isLast ? W - R : X(L, ow, i), T + oh + 18);
      }
      ctx.textAlign = "right"; ctx.fillStyle = "#333"; ctx.font = "12px sans-serif";
      ctx.fillText(`${xAxis} →`, W - R, H - 5);
    }

    // ---------- 下半：每个频率各自的波形 ----------
    {
      const L = 66, R = 16, gap = 14;
      const cw = (W - L - R - gap * (nCols - 1)) / nCols;
      const colors = ["#1f77b4", "#2ca02c", "#8a5fc5", "#17becf", "#d62728", "#bcbd22", "#e377c2", "#7f7f7f"];
      let allE = 0;
      for (const t of series.comps) allE += t.a * t.a + t.b * t.b;
      for (let j = 0; j < n; j++) {
        const t = topN[j];
        const col = j % nCols, row = (j / nCols) | 0;
        const ox = L + col * (cw + gap), oy = topH + 12 + row * cellH;
        const iT = 20, iB = 26, ox2 = ox + 12, ow2 = cw - 24, oh2 = cellH - iT - iB;

        const wv = new Float64Array(p);
        let m = 1e-12;
        for (let i = 0; i < p; i++) {
          const th = (2 * Math.PI * t.k * i) / p;
          wv[i] = t.a * Math.cos(th) + t.b * Math.sin(th);
          m = Math.max(m, Math.abs(wv[i]));
        }
        const Y = (v) => oy + iT + oh2 / 2 - ((oh2 / 2) * v) / m;
        const share = allE > 0 ? (100 * (t.a * t.a + t.b * t.b)) / allE : 0;

        ctx.strokeStyle = "#e8eaef"; ctx.lineWidth = 1;
        ctx.strokeRect(ox, oy, cw, cellH);
        ctx.strokeStyle = "#eef0f4";
        for (let g = 1; g < 4; g++) {
          const gy = oy + iT + (oh2 * g) / 4;
          ctx.beginPath(); ctx.moveTo(ox2, gy); ctx.lineTo(ox2 + ow2, gy); ctx.stroke();
        }
        ctx.strokeStyle = "#c9ced7"; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(ox2, Y(0)); ctx.lineTo(ox2 + ow2, Y(0)); ctx.stroke();
        ctx.setLineDash([]);

        ctx.strokeStyle = colors[j % colors.length]; ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let i = 0; i < p; i++) {
          const px = ox2 + (ow2 * i) / (p - 1), py = Y(wv[i]);
          i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        }
        ctx.stroke();

        ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 11.5px sans-serif"; ctx.textAlign = "left";
        ctx.fillText(`k = ${t.k}`, ox + 10, oy + 14);
        ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
        ctx.fillText(`R = ${t.amp.toFixed(3)} · 占能量 ${share.toFixed(1)}%`, ox + cw - 10, oy + 14);
        ctx.textAlign = "center"; ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif";
        ctx.fillText("0", ox2, oy + cellH - 8);
        ctx.fillText(String(p - 1), ox2 + ow2, oy + cellH - 8);
      }
    }
  }

  // ===================================================================
  // 频率视图：沿数据流逐级看同一件事 —— 某一维是不是少数几个 sin/cos 波
  //   EMBED → ATTENTION → MLP → LOGITS，正好是傅里叶乘法电路的全链路
  //   （读出权重 W_U 的视图已下线：它的“输出”就是 logits，想看权重看下面热图区）
  // ===================================================================
  const FREQ_SRC = {
    embed: {
      name: "EMBED 频率", icon: "🎼", btn: "embedBtn", stage: "embed",
      selLabel: "词嵌入维度", selKind: "dim", varName: "a",
      xAxis: "token a", qty: "W_E[a, c]",
      cap1: "① 把这一列看成 token 的函数：横轴 a = 0…P−1，纵轴 W<sub>E</sub>[a, c]（纯 embedding 矩阵，不含位置项）",
      where: "论文 3.1 第一句 / 图 3（左）",
      why: "<b>词嵌入把 token a 映射到 sin(w<sub>k</sub>a)、cos(w<sub>k</sub>a)</b>：每一维都只是少数几个 sin/cos 波的固定组合，而不是一张任意查表。",
      hint: "固定<b>一个词嵌入维</b>、扫 token a，把它拆成几个 sin/cos 波——上图看“少数几个波就能重建整条曲线”，下图看谱有多稀疏。" +
            "结构图上对应 EMBED 里那条 <b>W<sub>E</sub>[a]</b> 向量（a / b / = 三条中的第一条），只点亮它、不点亮整块 EMBED。",
      vec: "a",
    },
    attn: {
      name: "ATTENTION 频率", icon: "🎯", btn: "attnFreqBtn", stage: "attn",
      selLabel: "残差流维度", selKind: "dim", varName: "b",
      xAxis: "token b", qty: "attention 输出[=][c]",
      cap1: "① 固定 a = 0、扫 b：横轴 b = 0…P−1，纵轴 = 位置「=」处 4 个注意力头输出之和的第 c 维",
      where: "论文 4.2 / 图 4（左）",
      why: "注意力用三角恒等式把 sin/cos(w<sub>k</sub>a)、sin/cos(w<sub>k</sub>b) 相乘相加，合成 sin/cos(w<sub>k</sub>(a+b))——这一步之后还是同一批频率。",
      hint: "固定 a = 0、扫 b，看位置「=」上 4 个注意力头的输出之和：它已经在算 sin/cos(w<sub>k</sub>(a+b))，谱同样只有几个尖峰。" +
            "结构图上点亮的是 ATTENTION 里最下面那条 <b>Σ 4 头输出相加 → W<sub>O</sub> 投影</b>子层（4 个头的 out 由右侧汇聚线收进它），不是整个 ATTENTION 块。",
      part: "attnSum",
    },
    mlp: {
      name: "MLP 频率", icon: "🧩", btn: "mlpFreqBtn", stage: "mlp",
      selLabel: "残差流维度", selKind: "dim", varName: "b",
      xAxis: "token b", qty: "MLP out[c]",
      cap1: "① 固定 a = 0、扫 b：横轴 b = 0…P−1，纵轴 = MLP 输出（W<sub>out</sub>·ReLU(…)）的第 c 维",
      where: "论文 4.2 / 图 4（中）",
      why: "MLP 神经元被单一频率解释（论文图 4 中），它们加回残差流的输出也落在同一批频率上。",
      hint: "固定 a = 0、扫 b，看 MLP 加回残差流的那个 128 维向量（= W<sub>out</sub>·ReLU(…) + b<sub>out</sub>）——它和 attention 一样只由少数频率构成。" +
            "结构图上点亮的是 MLP 最右边那一列<b>输出向量</b>（加回残差流的 128 维增量），不是整个 MLP 块、也不是权重框；" +
            "W_in / W_out 只作为层次标注画在两侧。",
      part: "mlpOut",
    },
    logits: {
      name: "LOGITS 频率", icon: "📊", btn: "logitsFreqBtn", stage: "logits",
      selLabel: "输出类", selKind: "class", varName: "b",
      xAxis: "token b", qty: "logit_c(a=0, b)",
      cap1: "① 固定 a = 0、扫 b：横轴 b = 0…P−1，纵轴 = 所选输出类 c 的 logit",
      where: "论文 4.1（logits 的周期性）/ 图 4（右）",
      why: "最后的 logits 也只在关键频率上有分量：几个余弦波在这里<b>相长/相消干涉</b>，正确类别 c* = a+b 处被抬到最大。",
      hint: "固定 a = 0、扫 b，看某个输出类的 logit：全链路的频率在这里相长/相消干涉，正确类别被抬到最大。" +
            "结构图上点亮的是最右侧 logits 输出向量里<b>所选输出类的那一格</b>（换“输出类”下拉框会跟着移动）。" +
            "上面那个 UNEMBED 矩阵只作为结构示意（读出权重，不再单独出面板）；这里看的才是它真正读出来的值。",
      part: "logitsCell",
    },
  };

  /** 取 src 指定的那条长度 p 的曲线（sel = 维度下标或输出类）。 */
  function freqSignal(src, sel, p, d) {
    const y = new Float64Array(p);
    if (src === "embed") {
      const te = state.mats["token_embed"];                 // [(p+1), d]，行 = token
      for (let i = 0; i < p; i++) y[i] = te.data[i * d + sel];
    } else if (src === "logits") {
      for (let b = 0; b < p; b++) y[b] = forwardOne(0, b).logits[sel];
    } else {
      const key = src === "attn" ? "attn" : "mlp";
      for (let b = 0; b < p; b++) y[b] = forwardOne(0, b)[key][sel];
    }
    return y;
  }

  /** 稀疏度评分：前 6 个频率占总能量（去直流）的比例。 */
  function freqConcentration(y, p) {
    const s = fourierSeries(y, p);
    let kept = 0, all = 0;
    for (const t of s.comps) all += t.a * t.a + t.b * t.b;
    for (const t of topFreqs(s, 6)) kept += t.a * t.a + t.b * t.b;
    return all > 0 ? kept / all : 0;
  }

  /** 自动挑一条最“正弦”的维度 / 输出类（前 6 个频率占比最高）。 */
  function pickFreqIndex(src, p, d) {
    if (src === "embed") {                                  // 直接读矩阵，很便宜
      let best = 0, bs = -1;
      for (let k = 0; k < d; k++) {
        const sc = freqConcentration(freqSignal(src, k, p, d), p);
        if (sc > bs) { bs = sc; best = k; }
      }
      return best;
    }
    // 需要前向的类型：一趟扫 b，把所有维度/输出类的曲线一起收下来再评分
    const n = src === "logits" ? p : d;
    const cols = Array.from({ length: n }, () => new Float64Array(p));
    for (let b = 0; b < p; b++) {
      const f = forwardOne(0, b);
      const v = src === "logits" ? f.logits : (src === "attn" ? f.attn : f.mlp);
      for (let k = 0; k < n; k++) cols[k][b] = v[k];
    }
    let best = 0, bs = -1;
    for (let k = 0; k < n; k++) {
      const sc = freqConcentration(cols[k], p);
      if (sc > bs) { bs = sc; best = k; }
    }
    return best;
  }

  /** 频率视图主流程：src ∈ embed / attn / mlp / unembed / logits */
  async function runFreqView(src) {
    const spec = FREQ_SRC[src] || FREQ_SRC.embed;
    const btn = $(`#${spec.btn}`);
    if (!state.cfg || !state.epochs || state.epochs.length === 0) return;
    // 互斥：隐藏测试面板
    for (const id of ["#testResult"]) {
      const el = $(id); if (el) el.hidden = true;
    }
    state.freqSrc = src;
    setPanelInfo(`freq:${src}`);
    const ep = state.loadedEpoch != null ? state.loadedEpoch : state.epochs[state.epochs.length - 1];
    if (!state.mats) await loadWeights(ep);
    if (!state.mats) { alert("请先载入权重（点上方“载入权重”）"); return; }
    btn.disabled = true; btn.innerHTML = "分析中…";
    try {
      const p = state.cfg.p, d = state.cfg.d_model;
      // 维度 / 输出类选择器：换一级就重建选项，并自动挑一条最“正弦”的
      const sel = $("#freqSel");
      const kind = spec.selKind === "class" ? "class" : "dim";
      $("#freqSelLabel").textContent = spec.selLabel;
      $("#freqTitle").textContent = `${spec.icon} ${spec.name}`;
      if (sel.dataset.src !== src) {
        sel.innerHTML = "";
        const n = kind === "class" ? p : d;
        for (let i = 0; i < n; i++) sel.add(new Option(`${kind === "class" ? "类 " : "dim "}${i}`, String(i)));
        sel.dataset.src = src;
        if (state.freqPick[src] == null) state.freqPick[src] = pickFreqIndex(src, p, d);
        sel.value = String(state.freqPick[src]);
      }
      const topSel = $("#freqTop");
      if (topSel.options.length === 0) {
        for (let n = 1; n <= 8; n++) topSel.add(new Option(String(n), String(n)));
        topSel.value = "3";
      }
      const c = Number(sel.value), N = Number(topSel.value);

      // 维度 / 输出类选择器已经就绪，让结构图上的高亮再同步一次
      // （LOGITS 点亮的格子要跟 “输出类” 下拉框一致；setPanelInfo 调用时下拉框还没建好）
      highlightArch(spec.stage);

      const y = freqSignal(src, c, p, d);
      const series = fourierSeries(y, p);
      const topN = topFreqs(series, N);
      const keep = new Set(topN.map((t) => t.k));
      const yrec = rebuildFromFreqs(series, p, keep);

      let sse = 0, sst = 0;
      for (let n = 0; n < p; n++) {
        const e = y[n] - yrec[n]; sse += e * e;
        const t = y[n] - series.c0; sst += t * t;
      }
      const rms = Math.sqrt(sse / p);
      const errPct = sst > 0 ? (100 * sse) / sst : 0;
      let keptE = 0, allE = 0;
      for (const t of topN) keptE += t.a * t.a + t.b * t.b;
      for (const t of series.comps) allE += t.a * t.a + t.b * t.b;
      const sharePct = allE > 0 ? (100 * keptE) / allE : 0;

      $("#freqCap1").innerHTML = spec.cap1;
      $("#freqCap2").innerHTML = `② 频率谱：对 ${spec.xAxis} 做 DFT 后各频率 k 的幅度 |c<sub>k</sub>|（横轴 k = 1…P/2，红 = 保留）`;
      $("#freqCap3").innerHTML = "③ 拆解与叠加：上半 = 把保留的频率<strong>一条条加上去</strong>（浅 → 深，最后一条 = 重建）；下半 = 每条波单独长什么样";

      drawEmbedWave($("#cEmbedWave"), p, y, yrec, { c, rms, errPct, topN, qty: spec.qty, xAxis: spec.xAxis });
      drawEmbedSpec($("#cEmbedSpec"), p, series, keep);
      drawEmbedParts($("#cEmbedParts"), p, y, series, topN, yrec, spec.xAxis);

      const st = stageOf(ep);
      $("#freqStage").textContent = `${ep} 步${st ? " · " + st : ""}`;
      $("#edWaveNote").innerHTML =
        `上图这条 <b>${spec.qty}</b> 曲线只由 <b>${topN.length} 个频率</b>（k = ${topN.map((t) => t.k).join(", ")}，` +
        `每个频率含 1 个 cos 波 + 1 个 sin 波）相加得到，却复现了原始曲线的 <b>${(100 - errPct).toFixed(2)}%</b>。`;
      $("#edSpecNote").innerHTML =
        `共 ${series.K} 个频率，前 ${topN.length} 个就占了该曲线总能量的 <b>${sharePct.toFixed(2)}%</b>。${spec.why}` +
        `<br>对应位置：${spec.where}；w<sub>k</sub> = 2πk/P。换维度或换训练阶段（顿悟前应当更杂乱）可以对比。`;
      $("#edPartsNote").innerHTML =
        `橙色那条 = 直流 <b>c₀ = ${series.c0.toFixed(3)}</b>（该列均值，只负责整体上下平移）+ 上面 <b>${topN.length}</b> 条波逐点相加。` +
        `上半图里浅色曲线是“只加前 1 个、前 2 个…频率”的中间结果，每加一个就更贴近灰色原始曲线；` +
        `下半图是每条波单独的样子：<b>R·cos(2πk${spec.varName}/P − φ)</b>，幅度 R 与相位 φ 完全由权重决定（不是人挑的），` +
        `把它们相加就得到①里那条橙色重建曲线。`;
      $("#freqResult").hidden = false;
    } catch (err) {
      console.error(err);
      alert(`${spec.name}失败：${err.message}`);
    } finally {
      btn.disabled = false; btn.innerHTML = `${spec.icon} ${spec.name}`;
    }
  }

  async function onRun() {
    state.run = $("#runSel").value;
    const seeds = Object.keys(state.index.runs[state.run].seeds).sort((a, b) => Number(a) - Number(b));
    const ss = $("#seedSel");
    ss.innerHTML = "";
    seeds.forEach((sd) => ss.add(new Option(`seed ${sd}`, String(sd))));
    ss.onchange = onSeed;          // 切换 seed 立即重载对应快照与测试集
    onSeed();
  }

  async function init() {
    try {
      state.index = await fetchJSON("/viz-data/index.json");
    } catch (e) {
      $("#runSel").innerHTML = "<option>暂无 viz 数据（请先运行 python 导出）</option>";
      console.error(e);
      return;
    }
    const runs = Object.keys(state.index.runs);
    const rs = $("#runSel");
    runs.forEach((r) => {
      const seeds = Object.keys(state.index.runs[r].seeds).length;
      const label = r.length > 52 ? r.replace("mainline_p113_frac0.3_wd1_l1_d128_h4x32_m512_ep25000", "mainline(p113,λ=1)") : r;
      rs.add(new Option(`${label}  (${seeds} seeds)`, r));
    });
    // 只有一个 run 时隐藏“实验”选择（避免看似多余的控件），多个 run 时自动显示
    const rw = $("#runWrap");
    if (rw) rw.style.display = runs.length > 1 ? "" : "none";
    rs.onchange = onRun;
    $("#matSel").onchange = () => state.mats && matView();
    $("#headSel").onchange = () => state.mats && matView();
    const tb = $("#testBtn");
    if (tb) tb.addEventListener("click", runTest);
    // 五个“频率”按钮：同一套视图，只是数据来源不同（沿数据流 EMBED → … → LOGITS）
    for (const key of Object.keys(FREQ_SRC)) {
      const b = $(`#${FREQ_SRC[key].btn}`);
      if (b) b.addEventListener("click", () => runFreqView(key));
    }
    // 切换维度/输出类、保留频率数：面板正显示时立即重画
    const fsel = $("#freqSel");
    if (fsel) fsel.addEventListener("change", () => {
      const r = $("#freqResult");
      if (r && !r.hidden && state.freqSrc) runFreqView(state.freqSrc);
    });
    const ftop = $("#freqTop");
    if (ftop) ftop.addEventListener("change", () => {
      const r = $("#freqResult");
      if (r && !r.hidden && state.freqSrc) runFreqView(state.freqSrc);
    });
    onRun();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
