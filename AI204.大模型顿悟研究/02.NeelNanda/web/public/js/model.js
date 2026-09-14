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
    l3d: null,         // 3D 曲面 {as3,bs3,g3,rot,tilt}（用于拖拽视角重绘）
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
    const wIn = 210, wEmb = 140, wAtt = 270, wMlp = 330, wUn = 58;
    const inX = X0;
    const embX = inX + wIn + GAP;
    const attX = embX + wEmb + GAP;
    const mlpX = attX + wAtt + GAP;
    const unX = mlpX + wMlp + GAP;
    const inCx = inX + wIn / 2, embCx = embX + wEmb / 2, attCx = attX + wAtt / 2,
          mlpCx = mlpX + wMlp / 2, unCx = unX + wUn / 2;
    const AY = 310;                        // 箭头/数据流所在的水平线
    const W = unX + wUn + 40;              // 视口总宽（自动适配）

    let s = `<defs>
      <marker id="arr" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
        <path d="M0,0 L9,4.5 L0,9 Z" fill="#8b2f2f"/>
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
    const arrows = [[inX + wIn, embX], [embX + wEmb, attX], [attX + wAtt, mlpX], [mlpX + wMlp, unX]];
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
    s += `<g><rect x="${inX}" y="104" width="${wIn}" height="416" rx="8" fill="url(#bx)" stroke="#5b6172"/>`;
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

    // 2) EMBED
    s += `<g><rect x="${embX}" y="150" width="${wEmb}" height="320" rx="10" fill="url(#bx)" stroke="#c0843c"/>`;
    s += lbl(embCx, 140, "EMBED", 13, "#c9a15c");
    s += `<text x="${embCx}" y="300" fill="#dfe3ea" font-size="26" text-anchor="middle">W_E</text>`;
    s += lbl(embCx, 330, "one-hot → " + d + " 维", 12);
    s += lbl(embCx, 356, d + " × 3", 15, "#dfe3ea");
    s += `</g>`;

    // 3) ATTENTION（4 头）
    s += `<g><rect x="${attX}" y="70" width="${wAtt}" height="480" rx="12" fill="#1f2530" stroke="#7a5bb0"/>`;
    s += lbl(attCx, 55, `ATTENTION（${h} 头）`, 14, "#b18be0");
    const L = attX + 22;
    const HW = wAtt - 44;
    const headH = 106, top0 = 100, gapV = 12;
    for (let i = 0; i < h; i++) {
      const yy = top0 + i * (headH + gapV);
      s += `<rect x="${L}" y="${yy}" width="${HW}" height="${headH}" rx="8" fill="#2a3040" stroke="#5b6172"/>`;
      s += `<text x="${L + 14}" y="${yy + 18}" fill="#9aa3b2" font-size="10">head ${i}</text>`;
      const qx = L + 40, kx = L + 98, vx = L + 156, midY = yy + 52;
      s += `<circle cx="${qx}" cy="${midY}" r="10" fill="#3f4763" stroke="#8a90a3"/>
            <circle cx="${kx}" cy="${midY}" r="10" fill="#3f4763" stroke="#8a90a3"/>
            <circle cx="${vx}" cy="${midY}" r="10" fill="#3f4763" stroke="#8a90a3"/>`;
      s += `<text x="${qx}" y="${midY + 5}" fill="#dfe3ea" font-size="11" text-anchor="middle">Q</text>
            <text x="${kx}" y="${midY + 5}" fill="#dfe3ea" font-size="11" text-anchor="middle">K</text>
            <text x="${vx}" y="${midY + 5}" fill="#dfe3ea" font-size="11" text-anchor="middle">V</text>`;
      const mx = L + HW / 2, oy = midY + 26;
      s += `<line x1="${kx}" y1="${midY}" x2="${mx}" y2="${oy}" stroke="#b18be0" stroke-width="1.4"/>
            <line x1="${vx}" y1="${midY}" x2="${mx + 20}" y2="${oy}" stroke="#b18be0" stroke-width="1.4"/>`;
      s += `<circle cx="${mx}" cy="${oy}" r="11" fill="none" stroke="#e0c25a" stroke-width="1.6"/>`;
      s += `<text x="${mx}" y="${oy + 5}" fill="#e0c25a" font-size="13" text-anchor="middle">⊗</text>`;
      s += `<text x="${L + HW - 8}" y="${yy + 20}" fill="#9aa3b2" font-size="9" text-anchor="end">softmax(QKᵀ/√d)</text>`;
      s += `<rect x="${L + HW - 40}" y="${oy + 6}" width="34" height="16" rx="4" fill="#3f4763"/>
            <text x="${L + HW - 23}" y="${oy + 17}" fill="#dfe3ea" font-size="9" text-anchor="middle">out</text>`;
    }
    s += lbl(attCx, 580, "每个头：softmax(QKᵀ/√d_head)·V，再 W_O 投影求和；d_head=" + (cfg ? cfg.d_head : 32), 11);
    s += `</g>`;          // 关闭 ATTENTION 分组
    s += `</g>`;          // 关闭 translate（MLP 用绝对坐标、占满全高，下面单独画）

    // 4) MLP —— 圆圈=结点、线=权重；输入/隐藏/输出均按“单列(1D)”排布，结点数真实（128 / 512 / 128）
    //    boxY=70 / boxH=480：与上方 ATTENTION 完全等高、同顶（70→550）
    {
      const boxY = 70, boxH = 480;
      s += `<g><rect x="${mlpX}" y="${boxY}" width="${wMlp}" height="${boxH}" rx="12" fill="url(#bx)" stroke="#2e8b66"/>`;
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
      s += `<g stroke-linecap="round">${fan(xIn + 10, yIn, xHid - 10, yHid, "#8fd0ff")}${fan(xHid + 10, yHid, xOut - 10, yOut, "#a7e0b0")}</g>`;

      s += dotCol(xIn, yIn, 1.3, "#3f4763", "#9db4d0");
      s += dotCol(xHid, yHid, 0.6, "#1e7d46", "#8ae3a8");
      s += dotCol(xOut, yOut, 1.3, "#4a4f78", "#c9a15c");

      s += lbl((xIn + xHid) / 2, yTop - 16, "W_in", 10, "#8fd0ff");
      s += lbl((xHid + xOut) / 2, yTop - 16, "W_out", 10, "#a7e0b0");
      s += lbl(xIn, yBot + 22, `${d} 输入`, 10.5, "#dfe3ea");
      s += lbl(xHid, yBot + 22, `${m} 神经元 · ReLU`, 10.5, "#dfe3ea");
      s += lbl(xOut, yBot + 22, `${d} 输出`, 10.5, "#dfe3ea");
      s += lbl(mlpCx, boxY + boxH - 14, "（输入/输出 128 与隐藏 512 各为单列；512 在 480 高内偏密，连线为部分示意）", 9);
      s += `</g>`;
    }
    // 重新平移：画 UNEMBED 等
    s += `<g transform="translate(0,${VS})">`;
    // 5) UNEMBED + 输出
    s += `<g><rect x="${unX}" y="120" width="${wUn}" height="380" rx="8" fill="url(#bx)" stroke="#c0843c"/>`;
    s += lbl(unCx, 100, "UNEMBED", 11, "#c9a15c");
    for (let yy = 150; yy <= 470; yy += 46) {
      s += `<circle cx="${unCx}" cy="${yy}" r="5" fill="${yy === 340 ? '#d9a441' : '#556'}" stroke="#8a90a3"/>`;
    }
    s += lbl(unCx, 512, "logits", 12, "#dfe3ea");
    s += lbl(unCx, 534, p + " 类", 10);
    s += `</g>`;       // 关闭 UNEMBED 分组
    s += `</g>`;       // 关闭 translate 分组

    // 测试后的预测结果显示在输出端
    if (hasS && sample.pred != null) {
      const ok = !!sample.correct;
      s += `<text x="${unCx}" y="566" fill="${ok ? '#6ee7a0' : '#ff8a80'}" font-size="12.5" text-anchor="middle" font-family="sans-serif">ĉ = ${sample.pred}${ok ? " ✓" : " ✗"}</text>`;
    }

    svg.innerHTML = s;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
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

  /** 按新权重刷新仍在显示的临时结果（测试或 MLP 循环，二选一） */
  async function refreshPanels() {
    if (!state.mats) return;
    const lg = $("#logitsResult"), sw = $("#sweepResult"), tr = $("#testResult");
    if (lg && !lg.hidden) { await runLogitsSweep(); return; }
    if (sw && !sw.hidden) { await runMlpSweep(); return; }
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
      state.grid = null;            // 权重换了 → 全分辨率网格/显示层缓存作废
      state.heatField = null;
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
    // logits 输出类下拉（0…P−1），保留已有选择
    const lgs = $("#lgSel");
    if (lgs) {
      const cur = lgs.value;
      lgs.innerHTML = "";
      for (let c = 0; c < idx.config.p; c++) lgs.add(new Option(String(c), String(c)));
      lgs.value = (cur !== "" && Number(cur) < idx.config.p) ? cur : "0";
    }
    selectEpoch(state.epochs[state.epochs.length - 1]);
  }

  // ===================================================================
  // 测试：取一个测试样本 → 前向传播 → logits
  // ===================================================================
  function clearTest() {
    state.sample = null;
    state.testList = null;
    state.testKey = null;
    state.lastTest = null;
    const r = $("#testResult");
    if (r) r.hidden = true;
    const sw = $("#sweepResult");
    if (sw) sw.hidden = true;
    const lg = $("#logitsResult");
    if (lg) lg.hidden = true;
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

  /** 对外接口：返回本次前向的副本（可安全持有） */
  function forwardOne(a, b) {
    const ws = forwardCore(a, b);
    return { logits: ws.logits.slice(), h: ws.h.slice() };
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
    // 互斥：点“测试”时隐藏 MLP 循环与 LOGITS 面板
    const swX = $("#sweepResult");
    if (swX) swX.hidden = true;
    const lgX = $("#logitsResult");
    if (lgX) lgX.hidden = true;
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
  // MLP 循环：固定 a=0，扫 b=0…P-1；一次画出前 10 个神经元的输出（2 列 × 5 行）
  // ===================================================================
  function drawSweepGrid(canvas, p, rows, lbl = "n") {
    // rows[i][b] = 第 i 个单元在 (a=0, b) 的输出（lbl 用于小图标题前缀，如 n/c）
    const W = 1100, H = 770;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const n = rows.length;
    const cols = 2;
    const nRows = Math.ceil(n / cols);
    const L = 12, R = 12, T = 6, B = 34;
    const gapC = 24, gapR = 12;
    const cw = (W - L - R - gapC * (cols - 1)) / cols;
    const ch = (H - T - B - gapR * (nRows - 1)) / nRows;
    const mT = 18;                              // 单元内顶部（标题/幅值）留白
    const colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                    "#17becf", "#e377c2", "#bcbd22", "#7f7f7f", "#8c564b"];
    const fmtY = (v) => {
      if (v === 0) return "0";
      const a = Math.abs(v);
      if (a < 0.01) return v.toExponential(1);
      return v.toFixed(2);
    };
    ctx.font = "11px sans-serif";
    for (let k = 0; k < n; k++) {
      const c = k % cols, r = (k / cols) | 0;
      const ox = L + c * (cw + gapC), oy = T + r * (ch + gapR);
      const pw = cw - 14, ph = ch - mT - 6;
      const X = (b) => ox + 10 + (pw * b) / (p - 1);
      let vmax = -Infinity, vmin = Infinity;
      for (let b = 0; b < p; b++) {
        const v = rows[k][b];
        if (v > vmax) vmax = v;
        if (v < vmin) vmin = v;
      }
      if (vmin > 0) vmin = 0;                    // 全为正（如 ReLU 激活）：下限贴 0 基线
      if (!isFinite(vmin) || vmax <= vmin) { vmin = 0; vmax = Math.max(vmax, 1); }
      const Y = (v) => oy + mT + ph - (ph * (v - vmin)) / (vmax - vmin);
      const baseY = Y(0);
      // 单元边框 + 网格
      ctx.strokeStyle = "#e8eaef"; ctx.lineWidth = 1;
      ctx.strokeRect(ox, oy, cw, ch);
      ctx.strokeStyle = "#eef0f4";
      for (let g = 1; g < 4; g++) {
        const gy = oy + mT + (ph * g) / 4;
        ctx.beginPath(); ctx.moveTo(X(0), gy); ctx.lineTo(X(p - 1), gy); ctx.stroke();
      }
      ctx.strokeStyle = "#c9ced7"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(X(0), baseY); ctx.lineTo(X(p - 1), baseY); ctx.stroke();
      ctx.setLineDash([]);
      // 曲线
      ctx.strokeStyle = colors[k % colors.length]; ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let b = 0; b < p; b++) {
        const px = X(b), py = Y(rows[k][b]);
        b ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
      // 标题（单元下标）+ 最大/最小
      ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "left";
      ctx.fillText(`${lbl}${k}`, ox + 8, oy + 13);
      ctx.fillStyle = "#8a93a5"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
      const rangeTxt = fmtY(vmax) + (vmin < 0 ? " / " + fmtY(vmin) : "");
      ctx.fillText(rangeTxt, ox + cw - 8, oy + 13);
      // 仅最底行的 x 轴标 b
      if (r === nRows - 1) {
        ctx.fillStyle = "#8a93a5"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
        const tick = Math.max(1, Math.ceil(p / 8));
        for (let b = 0; b < p; b += tick) ctx.fillText(String(b), X(b), oy + ch + 12);
        ctx.fillText(String(p - 1), X(p - 1), oy + ch + 12);
      }
    }
    ctx.fillStyle = "#333"; ctx.font = "12px sans-serif"; ctx.textAlign = "right";
    ctx.fillText("b →", W - R - 2, H - 8);
  }

  async function runMlpSweep() {
    const btn = $("#mlpBtn");
    if (!state.cfg || !state.epochs || state.epochs.length === 0) return;
    // 互斥：点“MLP 循环”时隐藏测试与 LOGITS 面板
    const trX = $("#testResult");
    if (trX) trX.hidden = true;
    const lgX = $("#logitsResult");
    if (lgX) lgX.hidden = true;
    if (!state.mats) {
      const ep = state.loadedEpoch != null ? state.loadedEpoch : state.epochs[state.epochs.length - 1];
      await loadWeights(ep);
    }
    if (!state.mats) { alert("请先载入权重（点上方“载入权重”）"); return; }
    btn.disabled = true; btn.innerHTML = "扫描中…";
    try {
      const p = state.cfg.p;
      const N = Math.min(10, state.cfg.n_mlp || 512);   // 画前 10 个神经元
      const rows = Array.from({ length: N }, () => new Float64Array(p));
      for (let b = 0; b < p; b++) {
        const { h } = forwardOne(0, b);        // a=0 固定，b 扫描
        for (let i = 0; i < N; i++) rows[i][b] = h[i];
      }
      $("#swNeuron").textContent = `#0…#${N - 1}（前 ${N} 个）`;
      $("#swRange").textContent = `b = 0 … ${p - 1}（共 ${p} 个）`;
      const st = stageOf(state.loadedEpoch);
      $("#swStage").textContent = `${state.loadedEpoch} 步${st ? " · " + st : ""}`;
      $("#sweepResult").hidden = false;
      drawSweepGrid($("#cMlpSweep"), p, rows);
    } catch (err) {
      console.error(err);
      alert(`MLP 循环失败：${err.message}`);
    } finally {
      btn.disabled = false; btn.innerHTML = "🧠 MLP 循环";
    }
  }

  // ===================================================================
  // LOGITS 循环：固定 a=0，扫 b，画第 1 个 logits 单元（输出类 0）的单条大图
  // ===================================================================
  function drawSingleSweep(canvas, p, ys, label, xLabel = "b") {
    const W = 1100, H = 180;          // 高度减半（360 → 180）
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const L = 54, R = 14, T = 20, B = 30;
    const pw = W - L - R, ph = H - T - B;
    let vmin = Infinity, vmax = -Infinity;
    for (let b = 0; b < p; b++) { const v = ys[b]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    if (!isFinite(vmin)) { vmin = 0; vmax = 1; }
    const lo = Math.min(0, vmin), hi = Math.max(0, vmax);
    const lo2 = lo, hi2 = hi === lo ? hi + 1 : hi;
    const X = (b) => L + (pw * b) / (p - 1);
    const Y = (v) => T + ph - ((ph * (v - lo2)) / (hi2 - lo2));
    const fmt = (v) => { if (v === 0) return "0"; const a = Math.abs(v); if (a < 0.01) return v.toExponential(1); return v.toFixed(2); };
    ctx.strokeStyle = "#e6e8ee"; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) { const gy = T + (ph * g) / 4; ctx.beginPath(); ctx.moveTo(L, gy); ctx.lineTo(W - R, gy); ctx.stroke(); }
    ctx.strokeStyle = "#c9ced7"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(W - R, Y(0)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = "#8a5fc5"; ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let b = 0; b < p; b++) { const px = X(b), py = Y(ys[b]); b ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.stroke();
    ctx.fillStyle = "#57606a"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    for (let g = 0; g <= 4; g++) { const v = hi2 - ((hi2 - lo2) * g) / 4; ctx.fillText(fmt(v), L - 6, T + (ph * g) / 4 + 4); }
    ctx.textAlign = "center";
    const tick = Math.max(1, Math.ceil(p / 16));
    for (let b = 0; b < p; b += tick) ctx.fillText(String(b), X(b), H - B + 16);
    ctx.fillText(String(p - 1), X(p - 1), H - B + 16);
    ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 13px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(label, 6, 16);
    ctx.fillStyle = "#888"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    ctx.fillText(`${fmt(hi2)} / ${fmt(lo2)}`, W - R, 16);
    ctx.fillStyle = "#333"; ctx.font = "12px sans-serif"; ctx.textAlign = "right";
    ctx.fillText(`${xLabel} →`, W - R - 16, H - 6);
  }

  /** 3D 曲面：x=a、y=b、z=logit_c(a,b)。rot° 绕垂直轴旋转、tilt 控制竖直俯仰 */
  function drawLogits3D(canvas, as3, bs3, g3, rotDeg, tilt) {
    const W = 1100, H = 520;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const ny = as3.length, nx = bs3.length;
    const amax = as3[ny - 1], bmax = bs3[nx - 1];
    let zmin = Infinity, zmax = -Infinity;
    for (let i = 0; i < g3.length; i++) { const z = g3[i]; if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
    if (!isFinite(zmin)) { zmin = 0; zmax = 1; }
    const rng = zmax - zmin || 1;
    const c30 = Math.cos(Math.PI / 6);
    const th = rotDeg * Math.PI / 180, ct = Math.cos(th), st = Math.sin(th);
    // 屏幕基：a→右下方、b→左下方，绕屏幕法线转 rot°，tilt 压扁竖直方向
    const ea0x = c30, ea0y = 0.5, eb0x = -c30, eb0y = 0.5;
    const ea = { x: ea0x * ct - ea0y * st, y: (ea0x * st + ea0y * ct) * tilt };
    const eb = { x: eb0x * ct - eb0y * st, y: (eb0x * st + eb0y * ct) * tilt };
    const X = (a, b) => a * ea.x + b * eb.x;
    const Yb = (a, b) => a * ea.y + b * eb.y;
    const hh = 170;
    const Y = (a, b, z) => Yb(a, b) - ((z - zmin) / rng) * hh;
    // 自动缩放与居中（考虑 z 高度）
    let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
    const upd = (a, b, z) => { const x = X(a, b), y = Y(a, b, z); if (x < minx) minx = x; if (x > maxx) maxx = x; if (y < miny) miny = y; if (y > maxy) maxy = y; };
    for (const a of [0, amax]) for (const b of [0, bmax]) { upd(a, b, zmin); upd(a, b, zmax); }
    const spanX = (maxx - minx) || 1, spanY = (maxy - miny) || 1;
    const S = Math.min((W - 150) / spanX, (H - 110) / spanY, 2.9);
    const ox = (W - (minx + maxx) * S) / 2;
    const oy = (H - (miny + maxy) * S) / 2;
    if (state.l3d) state.l3d.lastView = { ea, eb, ox, oy, S, amax, bmax };   // 供悬停反投影
    const PX = (a, b, z) => [ox + X(a, b) * S, oy + Y(a, b, z) * S];
    // painter：按基面竖直坐标排序，远(小)先画
    const cells = [];
    for (let ia = 0; ia < ny - 1; ia++) for (let ib = 0; ib < nx - 1; ib++) {
      const a0 = as3[ia], a1 = as3[ia + 1], b0 = bs3[ib], b1 = bs3[ib + 1];
      const zm = (g3[ia * nx + ib] + g3[(ia + 1) * nx + ib] + g3[(ia + 1) * nx + ib + 1] + g3[ia * nx + ib + 1]) / 4;
      cells.push({ ia, ib, zm, d: Yb((a0 + a1) / 2, (b0 + b1) / 2) });
    }
    cells.sort((p, q) => p.d - q.d);
    const amp = Math.max(Math.abs(zmin), Math.abs(zmax)) || 1;
    const col = (v) => { const t = Math.min(1, Math.max(0, (v / amp + 1) / 2)); const rgb = divergingRGB(t); return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`; };
    for (const c of cells) {
      const a0 = as3[c.ia], a1 = as3[c.ia + 1], b0 = bs3[c.ib], b1 = bs3[c.ib + 1];
      const z00 = g3[c.ia * nx + c.ib], z10 = g3[(c.ia + 1) * nx + c.ib],
            z11 = g3[(c.ia + 1) * nx + c.ib + 1], z01 = g3[c.ia * nx + c.ib + 1];
      ctx.beginPath();
      ctx.moveTo(...PX(a0, b0, z00)); ctx.lineTo(...PX(a1, b0, z10));
      ctx.lineTo(...PX(a1, b1, z11)); ctx.lineTo(...PX(a0, b1, z01));
      ctx.closePath();
      ctx.fillStyle = col(c.zm); ctx.fill();
      ctx.strokeStyle = "rgba(25,28,38,0.18)"; ctx.lineWidth = 0.6; ctx.stroke();
    }
    // 坐标轴（地面 z=zmin）
    ctx.font = "13px sans-serif"; ctx.textAlign = "center";
    ctx.strokeStyle = "#555"; ctx.lineWidth = 1.6; ctx.fillStyle = "#333";
    const [x0, y0] = PX(0, 0, zmin);
    const [xa, ya] = PX(amax, 0, zmin);
    const [xb, yb2] = PX(0, bmax, zmin);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(xa, ya); ctx.stroke();
    ctx.fillText("a →", xa, ya + 16);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(xb, yb2); ctx.stroke();
    ctx.fillText("b →", xb, yb2 + 16);
    ctx.strokeStyle = "#c0392b";
    const [xza, yza] = PX(0, 0, zmax);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(xza, yza); ctx.stroke();
    ctx.fillStyle = "#333"; ctx.textAlign = "left"; ctx.font = "12px sans-serif";
    ctx.fillText(`z = logit_c，范围 ${zmin.toFixed(1)} … ${zmax.toFixed(1)} · 网格 ${nx}×${ny} · 拖动旋转视角`, 14, H - 10);
  }

  /** 按 state.l3d 里的视角重画 3D */
  function syncL3() {
    const d = state.l3d, cv = $("#cLogits3D");
    if (!d || !cv) return;
    drawLogits3D(cv, d.as3, d.bs3, d.g3, d.rot, d.tilt);
  }

  /** 给 3D 画布绑定一次交互：拖拽旋转/俯仰 + 悬停反投影显示 (a,b) */
  function attachL3(cv) {
    if (cv.__l3k) return;
    cv.__l3k = true;
    cv.style.cursor = "crosshair";
    cv.style.touchAction = "none";
    let drag = false, lx = 0, ly = 0;
    let tip = null, hoverKey = "", hoverZ = NaN;
    const makeTip = () => {
      if (tip) return tip;
      tip = document.createElement("div");
      tip.style.cssText = "position:fixed;pointer-events:none;background:rgba(12,16,24,.95);color:#fff;"
        + "font:12px/1.5 sans-serif;padding:4px 9px;border-radius:6px;border:1px solid #3a4458;"
        + "z-index:99999;white-space:nowrap;";
      document.body.appendChild(tip);
      return tip;
    };
    const hideTip = () => { if (tip) tip.style.display = "none"; };

    cv.addEventListener("pointerdown", (e) => {
      if (e.button === 0) { drag = true; lx = e.clientX; ly = e.clientY; hideTip(); cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing"; }
    });
    cv.addEventListener("pointermove", (e) => {
      const d = state.l3d;
      if (drag) {                      // 拖动：旋转 / 俯仰
        if (!d) return;
        d.rot = Math.max(-80, Math.min(80, d.rot + (e.clientX - lx) * 0.4));
        d.tilt = Math.max(0.35, Math.min(1.6, d.tilt - (e.clientY - ly) * 0.003));
        lx = e.clientX; ly = e.clientY;
        syncL3();
        return;
      }
      if (!d || !d.lastView || e.buttons !== 0) { hideTip(); return; }
      // 悬停：把光标反投影到 (a,b) 网格
      const V = d.lastView;
      const rect = cv.getBoundingClientRect();
      const ix = (e.clientX - rect.left) * (cv.width / rect.width);
      const iy = (e.clientY - rect.top) * (cv.height / rect.height);
      const u = (ix - V.ox) / V.S, v = (iy - V.oy) / V.S;
      const det = V.ea.x * V.eb.y - V.eb.x * V.ea.y;
      if (Math.abs(det) < 1e-9) { hideTip(); return; }
      const aI = Math.round((u * V.eb.y - V.eb.x * v) / det);
      const bI = Math.round((V.ea.x * v - u * V.ea.y) / det);
      if (aI < 0 || aI > V.amax || bI < 0 || bI > V.bmax) { hideTip(); return; }
      const key = aI + "," + bI;
      if (key !== hoverKey) {          // 只在格点变化时才算一次前向
        hoverKey = key;
        try {
          const idx = Number(($("#lgSel") && $("#lgSel").value) || 0);
          const { logits } = forwardOne(aI, bI);
          hoverZ = logits[idx];
        } catch (err) { hoverZ = NaN; }
      }
      const t = makeTip();
      t.textContent = `a = ${aI} · b = ${bI}   z(logit_c) = ${Number.isFinite(hoverZ) ? hoverZ.toFixed(2) : "—"}`;
      t.style.left = (e.clientX + 16) + "px";
      t.style.top = (e.clientY + 14) + "px";
      t.style.display = "block";
    });
    const stop = () => { drag = false; cv.style.cursor = "crosshair"; hideTip(); };
    cv.addEventListener("pointerup", stop);
    cv.addEventListener("pointercancel", stop);
    cv.addEventListener("pointerleave", stop);
  }

  // ===================================================================
  // 2D 热图：全分辨率 P×P 逐点前向（不再用粗网格插值——粗网格步长与 P 不互质时
  // 会整体漏掉只有 1 格宽的对角线）。显示量可选：softmax 概率 / 相对 logit / 原始 logit。
  // ===================================================================
  const HEAT = { W: 600, H: 546, L: 52, R: 12, T: 18, B: 58 };

  /** 全分辨率 logit 网格 Z[(a*P+b)*P+c]，按 (run, seed, epoch) 缓存（约 1.3 万次前向） */
  async function ensureFullGrid() {
    const p = state.cfg.p;
    const key = `${state.run}#${state.seed}#${state.loadedEpoch}`;
    if (state.grid && state.grid.key === key) return state.grid;
    const gen = state.gen;
    const prog = $("#lgProg");
    const Z = new Float32Array(p * p * p);
    const t0 = performance.now();
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) Z.set(forwardCore(a, b).logits, (a * p + b) * p);
      if (a % 2 === 1) {
        if (prog) prog.textContent = `计算中 ${Math.round(((a + 1) / p) * 100)}%`;
        await new Promise((r) => setTimeout(r, 0));     // 让出主线程，避免卡界面
        if (gen !== state.gen) return null;             // 阶段/seed 已切换，放弃这次
      }
    }
    const ms = performance.now() - t0;
    if (prog) prog.textContent = `${p}×${p} · ${(ms / 1000).toFixed(1)} s`;
    state.grid = { key, p, Z, ms };
    return state.grid;
  }

  /** 从网格抽出「显示量」层；同时存下 p(c) 与 argmax 供悬停提示 */
  function buildHeatField(grid, idx, qty) {
    const p = grid.p, Z = grid.Z;
    const F = new Float32Array(p * p), P = new Float32Array(p * p), MX = new Int16Array(p * p);
    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) {
        const off = (a * p + b) * p, k = a * p + b;
        let mx = -Infinity, sum = 0, am = 0;
        for (let c = 0; c < p; c++) { const t = Z[off + c]; sum += t; if (t > mx) { mx = t; am = c; } }
        let es = 0;
        for (let c = 0; c < p; c++) es += Math.exp(Z[off + c] - mx);
        const z = Z[off + idx];
        P[k] = Math.exp(z - mx) / es;
        MX[k] = am;
        F[k] = qty === "prob" ? P[k] : (qty === "rel" ? z - sum / p : z);
      }
    }
    return { key: `${grid.key}#${idx}#${qty}`, p, idx, qty, F, P, MX };
  }

  /** 顺序色（白 → 深红），给概率用：0 就是白，越接近 1 越红 */
  function seqRedRGB(t) {
    const u = Math.min(1, Math.max(0, t));
    return [Math.round(255 - 100 * u), Math.round(255 - 237 * u), Math.round(255 - 223 * u)];
  }

  const HEAT_NOTE = {
    prob: "softmax 概率 p(c|a,b)：只有 a+b ≡ c (mod P) 的位置 ≈ 1（红），其余 ≈ 0（白）——这才是「模型选中类 c」的位置。",
    rel: "相对 logit = logit_c −（该点各类 logit 的均值）：去掉了每条输入的公共分量，对角带最红；背景红蓝条纹来自模型所用的稀疏傅里叶频率（见 key_freqs.json）。",
    raw: "原始 logit：模型对某个固定类别的 logit 本身随 a+b 快速振荡（关键频率 5/10/15/31/48），所以整幅是周期性条纹而非一条对角线；又因多数位置为正，以 0 为白时整体偏红。",
  };

  /** 2D 热图：x=a、y=b、逐格近邻采样（真实值，不做插值） */
  function drawLogitsHeat(canvas, f) {
    const { W, H, L, R, T, B } = HEAT;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const p = f.p;
    let amp = 1;
    if (f.qty !== "prob") {                 // 发散色以 0 为中心，取对称幅度
      amp = 0;
      for (let i = 0; i < f.F.length; i++) { const v = Math.abs(f.F[i]); if (v > amp) amp = v; }
      amp = amp || 1;
    }
    const pw = W - L - R, ph = H - T - B;
    const img = ctx.createImageData(pw, ph), dat = img.data;
    for (let iy = 0; iy < ph; iy++) {
      const b = Math.min(p - 1, Math.floor((iy / ph) * p));
      for (let ix = 0; ix < pw; ix++) {
        const a = Math.min(p - 1, Math.floor((ix / pw) * p));
        const v = f.F[a * p + b];
        const rgb = f.qty === "prob" ? seqRedRGB(v) : divergingRGB((v / amp + 1) / 2);
        const o = (iy * pw + ix) * 4;
        dat[o] = rgb[0]; dat[o + 1] = rgb[1]; dat[o + 2] = rgb[2]; dat[o + 3] = 255;
      }
    }
    ctx.putImageData(img, L, T);
    ctx.strokeStyle = "#cfd3da"; ctx.lineWidth = 1;
    ctx.strokeRect(L, T, pw, ph);
    const amax = p - 1, bmax = p - 1;
    const step = Math.max(1, Math.round(amax / 8));
    ctx.fillStyle = "#57606a"; ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    for (let v = 0; v <= amax; v += step) ctx.fillText(String(v), L + (v / amax) * pw, H - B + 14);
    ctx.textAlign = "right";
    for (let v = 0; v <= bmax; v += step) ctx.fillText(String(v), L - 5, T + (v / bmax) * ph + 3);
    ctx.fillStyle = "#333"; ctx.font = "12px sans-serif";
    ctx.textAlign = "right"; ctx.fillText("b ↓", L - 8, T - 6);
    ctx.textAlign = "left"; ctx.fillText("a →", W - R - 6, H - B + 30);
    // 色阶条画在图下方（不遮挡数据）
    const bx = L, bw = 140, by = H - B + 26, bh = 8;
    const grad = ctx.createLinearGradient(bx, 0, bx + bw, 0);
    if (f.qty === "prob") {
      grad.addColorStop(0, "#ffffff"); grad.addColorStop(1, "#9b1220");
    } else {
      grad.addColorStop(0, "#2166ac"); grad.addColorStop(0.5, "#ffffff"); grad.addColorStop(1, "#b2182b");
    }
    ctx.fillStyle = grad; ctx.fillRect(bx, by, bw, bh);
    ctx.strokeStyle = "#999"; ctx.strokeRect(bx, by, bw, bh);
    ctx.fillStyle = "#57606a"; ctx.font = "9px sans-serif";
    ctx.textAlign = "left"; ctx.fillText(f.qty === "prob" ? "0" : (-amp).toFixed(0), bx - 2, by + bh + 10);
    ctx.textAlign = "center"; ctx.fillText(f.qty === "prob" ? "0.5" : "0", bx + bw / 2, by + bh + 10);
    ctx.textAlign = "right"; ctx.fillText(f.qty === "prob" ? "1" : amp.toFixed(0), bx + bw + 2, by + bh + 10);
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 11px sans-serif";
    const ttl = f.qty === "prob" ? `p(c=${f.idx} | a, b)`
      : (f.qty === "rel" ? `logit_${f.idx} − mean_c logit_c` : `logit_${f.idx}(a, b)`);
    ctx.fillText(ttl, bx + bw + 14, by + bh);
    canvas.__hf = f;                        // 供悬停提示使用（与实际画出来的一致）
  }

  /** 取（或算）全分辨率网格 → 建显示层 → 画图 + 说明文字 */
  async function refreshHeat() {
    const cv = $("#cLogitsHeat");
    if (!cv || !state.mats || !state.cfg) return;
    const grid = await ensureFullGrid();
    if (!grid) return;
    const idx = Number(($("#lgSel") && $("#lgSel").value) || 0);
    const qty = ($("#lgQty") && $("#lgQty").value) || "prob";
    const fk = `${grid.key}#${idx}#${qty}`;
    if (!state.heatField || state.heatField.key !== fk) state.heatField = buildHeatField(grid, idx, qty);
    drawLogitsHeat(cv, state.heatField);
    attachHeat(cv);
    const note = $("#heatNote");
    if (note) note.textContent = HEAT_NOTE[qty] || "";
  }

  /** 给 2D 热图绑定悬停：反算 (a,b)，显示 a、b、(a+b) mod P、logit_c、p(c) 与 argmax */
  function attachHeat(cv) {
    if (cv.__hk) return;
    cv.__hk = true;
    cv.style.cursor = "crosshair";
    let tip = null, key = "";
    const make = () => {
      if (tip) return tip;
      tip = document.createElement("div");
      tip.style.cssText = "position:fixed;pointer-events:none;background:rgba(12,16,24,.95);color:#fff;"
        + "font:12px/1.5 sans-serif;padding:4px 9px;border-radius:6px;border:1px solid #3a4458;"
        + "z-index:99999;white-space:nowrap;";
      document.body.appendChild(tip);
      return tip;
    };
    const hide = () => { if (tip) tip.style.display = "none"; };
    cv.addEventListener("pointermove", (e) => {
      const f = cv.__hf;
      if (!f || e.buttons !== 0) { hide(); return; }
      const p = f.p, { L, R, T, B } = HEAT;
      const rect = cv.getBoundingClientRect();
      const ix = (e.clientX - rect.left) * (cv.width / rect.width);
      const iy = (e.clientY - rect.top) * (cv.height / rect.height);
      const pw = cv.width - L - R, ph = cv.height - T - B;
      if (ix < L || ix > L + pw || iy < T || iy > T + ph) { hide(); return; }
      const aI = Math.min(p - 1, Math.floor(((ix - L) / pw) * p));
      const bI = Math.min(p - 1, Math.floor(((iy - T) / ph) * p));
      const k = aI + "," + bI;
      const t = make();
      if (k !== key) {                    // 只在格子变化时更新内容
        key = k;
        const zc = state.grid ? state.grid.Z[(aI * p + bI) * p + f.idx] : NaN;
        t.textContent = `a = ${aI} · b = ${bI} · (a+b) mod P = ${(aI + bI) % p}`
          + ` · logit_c = ${Number.isFinite(zc) ? zc.toFixed(2) : "—"}`
          + ` · p(c) = ${f.P[aI * p + bI].toFixed(4)} · argmax = ${f.MX[aI * p + bI]}`;
      }
      t.style.left = (e.clientX + 16) + "px";
      t.style.top = (e.clientY + 14) + "px";
      t.style.display = "block";
    });
    const stop = () => { if (tip) tip.style.display = "none"; };
    cv.addEventListener("pointerleave", stop);
  }

  async function runLogitsSweep() {
    const btn = $("#logitsBtn");
    if (!state.cfg || !state.epochs || state.epochs.length === 0) return;
    // 互斥：点“LOGITS 循环”时隐藏测试与 MLP 面板
    const tR = $("#testResult");
    if (tR) tR.hidden = true;
    const sR = $("#sweepResult");
    if (sR) sR.hidden = true;
    if (!state.mats) {
      const ep = state.loadedEpoch != null ? state.loadedEpoch : state.epochs[state.epochs.length - 1];
      await loadWeights(ep);
    }
    if (!state.mats) { alert("请先载入权重（点上方“载入权重”）"); return; }
    btn.disabled = true; btn.innerHTML = "扫描中…";
    try {
      const p = state.cfg.p;
      const idx = Number(($("#lgSel") && $("#lgSel").value) || 0);   // 下拉选择的输出类
      const ysB = new Float64Array(p);   // ① 固定 a=0，扫 b
      for (let b = 0; b < p; b++) {
        const { logits } = forwardOne(0, b);
        ysB[b] = logits[idx];
      }
      const ysA = new Float64Array(p);   // ② 固定 b=0，扫 a
      for (let a = 0; a < p; a++) {
        const { logits } = forwardOne(a, 0);
        ysA[a] = logits[idx];
      }
      $("#lgRange").textContent = `b = 0 … ${p - 1}（共 ${p} 个）`;
      const st = stageOf(state.loadedEpoch);
      $("#lgStage").textContent = `${state.loadedEpoch} 步${st ? " · " + st : ""}`;
      $("#logitsResult").hidden = false;
      drawSingleSweep($("#cLogitsSweep"), p, ysB, `① c${idx} · logit（固定 a=0，扫 b）`, "b");
      const cA = $("#cLogitsSweepA");
      if (cA) drawSingleSweep(cA, p, ysA, `② c${idx} · logit（固定 b=0，扫 a）`, "a");
      // ③ 三维曲面：采样 ~29×29 的 (a,b) 网格
      const step = Math.max(1, Math.round((p - 1) / 28));
      const as3 = [], bs3 = [];
      for (let a = 0; a < p; a += step) as3.push(a);
      if (as3[as3.length - 1] !== p - 1) as3.push(p - 1);
      for (let b = 0; b < p; b += step) bs3.push(b);
      if (bs3[bs3.length - 1] !== p - 1) bs3.push(p - 1);
      const g3 = new Float64Array(as3.length * bs3.length);
      for (let ia = 0; ia < as3.length; ia++) {
        for (let ib = 0; ib < bs3.length; ib++) {
          const { logits } = forwardOne(as3[ia], bs3[ib]);
          g3[ia * bs3.length + ib] = logits[idx];
        }
      }
      const c3 = $("#cLogits3D");
      if (c3) {
        if (!state.l3d) state.l3d = { rot: 0, tilt: 1 };   // 保留用户视角；网格换成新的
        state.l3d.as3 = as3; state.l3d.bs3 = bs3; state.l3d.g3 = g3;
        attachL3(c3);
        syncL3();
      }
      const ch = $("#cLogitsHeat");
      if (ch) await refreshHeat();          // 全分辨率逐点重算（按 epoch 缓存）
    } catch (err) {
      console.error(err);
      alert(`LOGITS 循环失败：${err.message}`);
    } finally {
      btn.disabled = false; btn.innerHTML = "🔀 LOGITS 循环";
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
    const mb = $("#mlpBtn");
    if (mb) mb.addEventListener("click", runMlpSweep);
    const lb = $("#logitsBtn");
    if (lb) lb.addEventListener("click", runLogitsSweep);
    // 切换下拉的输出类：若 LOGITS 面板正显示则立即按新选择重画
    const lgs = $("#lgSel");
    if (lgs) lgs.addEventListener("change", () => {
      const r = $("#logitsResult");
      if (r && !r.hidden) runLogitsSweep();
    });
    // 切换 2D 热图的显示量（概率 / 相对 logit / 原始 logit）：网格已缓存，重画很快
    const lgq = $("#lgQty");
    if (lgq) lgq.addEventListener("change", () => {
      const r = $("#logitsResult");
      if (r && !r.hidden) refreshHeat();
    });
    onRun();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
