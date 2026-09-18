/* 「逆向工程：从数字到答案」页面的逻辑。
 *
 * 一半是教科书式的理想化示意（DFT 嵌入、旋转相乘、均匀频率打分），
 * 一半是**用真实权重现算**：直接从 viz-data 的快照里读出 W_E / W_U / 注意力 / MLP，
 * 在浏览器里跑一次前向（与 python/grokking/model.py 的 forward_full() 一致），
 * 并把「真实网络只用了哪几个频率」标出来——这正是论文逆向工程的核心结论。
 *
 * 快照格式（GKV1）与 model.js 里的 parseBin 一致；前向也与时/空一致，避免两处口径不同。
 */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // ---------------- 状态 ----------------
  const st = {
    P: 113, d: 128, nh: 4, dh: 32, nm: 512,
    run: null, seed: 0, epoch: null,
    mats: null, splitBytes: null, splitMeta: null,
    a: 15, b: 27,
    keys: [],            // 该网络真正用到的关键频率 k（从 W_E 的频谱里挑出来）
    dims: [],            // [{c, share}] 按重建占比降序
    c: 29,               // 当前查看的维度（W_E 的第 c 列）
    sim: { a: false, b: false },   // ③a 图上两个图标：用理想频率叠加还原 one-hot
    sim2: { a: false, b: false },  // ③c 图上两个图标：用真实 W_E 的频率做逆变换
    logits: null,        // 当前 (a,b) 的真实 logits
  };

  // ---------------- 快照解析（与 model.js 同一套） ----------------
  function parseBin(buf) {
    const dv = new DataView(buf);
    let magic = "";
    for (let i = 0; i < 4; i++) magic += String.fromCharCode(dv.getUint8(i));
    if (magic !== "GKV1") throw new Error("不是 GKV1 快照");
    const hlen = dv.getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen)));
    // 头部长度未必是 4 的倍数，先 slice 出一块独立缓冲区，保证 float32 起点对齐
    const f32 = new Float32Array(buf.slice(8 + hlen));
    const mats = {};
    for (const m of header.mats) {
      mats[m.name] = { shape: m.shape, data: f32.slice(m.start, m.start + m.numel) };
    }
    return { header, mats };
  }

  // ---------------- 傅里叶（实数序列的 cos/sin 系数） ----------------
  /** 返回 { c0, comps:[{k,a,b,amp}] }，k = 1…⌊P/2⌋ */
  function fourier(y, P) {
    let c0 = 0;
    for (let n = 0; n < P; n++) c0 += y[n];
    c0 /= P;
    const comps = [];
    for (let k = 1; k <= Math.floor(P / 2); k++) {
      let sa = 0, sb = 0;
      for (let n = 0; n < P; n++) {
        const th = (2 * Math.PI * k * n) / P;
        sa += y[n] * Math.cos(th);
        sb += y[n] * Math.sin(th);
      }
      const a = (2 / P) * sa, b = (2 / P) * sb;
      comps.push({ k, a, b, amp: Math.hypot(a, b) });
    }
    return { c0, comps };
  }

  /** 只用给定的频率集合重建 */
  function rebuild(series, P, keep) {
    const out = new Float64Array(P).fill(series.c0);
    for (const c of series.comps) {
      if (!keep.has(c.k)) continue;
      for (let n = 0; n < P; n++) {
        const th = (2 * Math.PI * c.k * n) / P;
        out[n] += c.a * Math.cos(th) + c.b * Math.sin(th);
      }
    }
    return out;
  }

  function shareOf(y, P, series, keep) {
    const rec = rebuild(series, P, keep);
    let sse = 0, sst = 0;
    for (let n = 0; n < P; n++) {
      sse += (y[n] - rec[n]) ** 2;
      sst += (y[n] - series.c0) ** 2;
    }
    return sst > 0 ? 100 * (1 - sse / sst) : 100;
  }

  /** 从 W_E 的频谱里挑出网络真正用到的关键频率：对每个 k 把所有维度的能量加起来 */
  function keyFreqsOf(WE, P, d, topN) {
    const energy = new Float64Array(P);
    for (let c = 0; c < d; c++) {
      const col = new Float64Array(P);
      for (let a = 0; a < P; a++) col[a] = WE[a * d + c];
      const s = fourier(col, P);
      for (const t of s.comps) energy[t.k] += t.a * t.a + t.b * t.b;
    }
    const rank = [];
    for (let k = 1; k <= Math.floor(P / 2); k++) rank.push({ k, e: energy[k] });
    rank.sort((x, y) => y.e - x.e);
    return rank.slice(0, topN).map((t) => t.k).sort((x, y) => x - y);
  }

  // ---------------- 前向传播（只要 logits；对齐 forward_full） ----------------
  function forward(a, b) {
    const { P, d, dh, nh, nm, mats } = st;
    const M = (n) => (mats[n] ? mats[n].data : null);
    const te = M("token_embed"), pe = M("pos_embed"), U = M("unembed");
    const x0 = new Float64Array(3 * d);
    const tok = [a, b, P];
    for (let t = 0; t < 3; t++) {
      const ot = tok[t] * d;
      for (let c = 0; c < d; c++) x0[t * d + c] = te[ot + c] + pe[t * d + c];
    }
    const acc = new Float64Array(3 * d);
    const invS = 1 / Math.sqrt(dh);
    for (let hi = 0; hi < nh; hi++) {
      const WQ = M(`head${hi}.W_Q`), WK = M(`head${hi}.W_K`), WV = M(`head${hi}.W_V`), WO = M(`head${hi}.W_O`);
      const q = new Float64Array(3 * dh), k = new Float64Array(3 * dh), v = new Float64Array(3 * dh);
      for (let t = 0; t < 3; t++) for (let j = 0; j < dh; j++) {
        let sq = 0, sk = 0, sv = 0;
        for (let c = 0; c < d; c++) {
          const xv = x0[t * d + c];
          sq += xv * WQ[j * d + c]; sk += xv * WK[j * d + c]; sv += xv * WV[j * d + c];
        }
        q[t * dh + j] = sq; k[t * dh + j] = sk; v[t * dh + j] = sv;
      }
      const attn = new Float64Array(9);
      for (let t = 0; t < 3; t++) {
        let mx = -Infinity;
        for (let t2 = 0; t2 < 3; t2++) {
          let s = 0;
          for (let j = 0; j < dh; j++) s += q[t * dh + j] * k[t2 * dh + j];
          s *= invS; attn[t * 3 + t2] = s; if (s > mx) mx = s;
        }
        let es = 0;
        for (let t2 = 0; t2 < 3; t2++) { const e = Math.exp(attn[t * 3 + t2] - mx); attn[t * 3 + t2] = e; es += e; }
        for (let t2 = 0; t2 < 3; t2++) attn[t * 3 + t2] /= es;
      }
      const ctx = new Float64Array(3 * dh);
      for (let t = 0; t < 3; t++) for (let j = 0; j < dh; j++) {
        let s = 0;
        for (let t2 = 0; t2 < 3; t2++) s += attn[t * 3 + t2] * v[t2 * dh + j];
        ctx[t * dh + j] = s;
      }
      for (let t = 0; t < 3; t++) for (let c = 0; c < d; c++) {
        let s = 0;
        for (let j = 0; j < dh; j++) s += ctx[t * dh + j] * WO[c * dh + j];
        acc[t * d + c] += s;
      }
    }
    const x = new Float64Array(3 * d);
    for (let i = 0; i < 3 * d; i++) x[i] = x0[i] + acc[i];
    const resid2 = new Float64Array(d);
    for (let c = 0; c < d; c++) resid2[c] = x[2 * d + c];
    const Wi = M("mlp_in.w"), bi = M("mlp_in.b"), Wo = M("mlp_out.w"), bo = M("mlp_out.b");
    const h = new Float64Array(nm);
    for (let j = 0; j < nm; j++) {
      let s = bi ? bi[j] : 0;
      for (let c = 0; c < d; c++) s += Wi[j * d + c] * resid2[c];
      h[j] = s > 0 ? s : 0;
    }
    const o = new Float64Array(d);
    for (let c = 0; c < d; c++) {
      let s = bo ? bo[c] : 0;
      for (let j = 0; j < nm; j++) s += Wo[c * nm + j] * h[j];
      o[c] = s;
    }
    const x2 = new Float64Array(d);
    for (let c = 0; c < d; c++) x2[c] = resid2[c] + o[c];
    const logits = new Float64Array(P);
    for (let r = 0; r < P; r++) {
      let s = 0;
      for (let c = 0; c < d; c++) s += U[r * d + c] * x2[c];
      logits[r] = s;
    }
    return logits;
  }

  // ---------------- 画布小工具 ----------------
  function ctxOf(cv) { return cv.getContext("2d"); }
  function clear(ctx, cv, bg) {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (bg) { ctx.fillStyle = bg; ctx.fillRect(0, 0, cv.width, cv.height); }
  }
  const FONT_S = "20px ui-monospace, SFMono-Regular, Menlo, monospace";
  const FONT_B = "bold 22px ui-monospace, SFMono-Regular, Menlo, monospace";

  // ---------------- ① 输入数字 ----------------
  function drawNumbers() {
    const P = st.P;
    const ans = (st.a + st.b) % P;
    $("#aVal").textContent = st.a;
    $("#bVal").textContent = st.b;
    $("#aBig").textContent = st.a;
    $("#bBig").textContent = st.b;
    $("#ansBig").textContent = ans;
    const pred = st.logits ? st.logits.indexOf(Math.max(...st.logits)) : null;
    $("#predBig").textContent = pred == null ? "—" : pred;
    const box = $("#predBox");
    if (pred != null) {
      const ok = pred === ans;
      box.className = "num-box " + (ok ? "green" : "red");
      $("#step1Note").innerHTML = ok
        ? `真实前向：<b>ĉ = ${pred}</b>，与真值一致 ✓（用的是 ${st.seedLabel} 的权重，逐点复算）`
        : `真实前向：<b>ĉ = ${pred}</b>，与真值 ${ans} 不一致 ✗（一般是还没「顿悟」的早期权重）`;
    }
  }

  // ---------------- ② one-hot ----------------
  function drawOneHot() {
    const cv = $("#onehotCanvas"), ctx = ctxOf(cv);
    const P = st.P;
    clear(ctx, cv, "#0e1016");
    const padL = 70, padR = 40, W = cv.width;
    const plotW = W - padL - padR;
    const bw = plotW / P;
    const rows = [
      { v: st.a, y: 46, h: 64, col: "#4ea1ff", label: "a" },
      { v: st.b, y: 170, h: 64, col: "#ff7eb6", label: "b" },
    ];
    for (const r of rows) {
      for (let n = 0; n < P; n++) {
        if (n === r.v) continue;
        ctx.fillStyle = "rgba(120,140,180,.28)";
        ctx.fillRect(padL + n * bw, r.y + r.h - 3, Math.max(bw - 0.6, 1), 3);
      }
      ctx.fillStyle = r.col;
      ctx.fillRect(padL + r.v * bw, r.y, Math.max(bw, 3), r.h);
      ctx.font = FONT_B;
      ctx.fillStyle = r.col;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(r.label, padL - 14, r.y + r.h / 2);
      ctx.font = FONT_S;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${r.label}[${r.v}] = 1`, clamp(padL + r.v * bw + bw / 2, 60, W - 60), r.y - 6);
    }
    // x 轴
    ctx.font = FONT_S;
    ctx.fillStyle = "#5d6a8f";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let n = 0; n <= P; n += 20) ctx.fillText(String(n % P), padL + n * bw, 250);
  }

  /**
   * 「几个频率叠加 → 还原出一个唯一波峰」面板的通用画法（③a 与 ③c 两块画布共用）。
   * o = { top, padL, plotW, P, t, col, waves, sum, sumW, capLeft, sideTop, sideBottom, normIdeal }
   *   waves : 每个频率各自的贡献（数组的数组，已含权重）
   *   sum   : 它们的叠加曲线（粗线）
   *   sumW  : 可选，按真实能量加权的叠加曲线
   */
  function drawPeakPanel(ctx, o) {
    const { top, padL, plotW, P, waves, sum } = o;
    const KH = Math.floor(P / 2);
    const pTop = top + 28, pH = 120, pBot = pTop + pH;
    const X = (n) => padL + (plotW * n) / (P - 1);

    let lo = 0, hi = 0;
    for (let n = 0; n < P; n++) { hi = Math.max(hi, sum[n]); lo = Math.min(lo, sum[n]); }
    if (!(hi > 0)) hi = 1;
    // 教科书版：1 个直流 + 全部 56 个 cos/sin 对 → 精确还原成一个尖峰（其余位置为 0）
    const ideal = new Float64Array(P);
    for (let n = 0; n < P; n++) {
      let v = 1 / P;
      for (let k = 1; k <= KH; k++) v += (2 / P) * Math.cos((2 * Math.PI * k * (o.t - n)) / P);
      ideal[n] = v * (o.normIdeal ? hi : 1);
    }
    const vMax = hi + (hi - lo) * 0.08, vMin = lo - (hi - lo) * 0.08;
    const Y = (v) => pTop + ((vMax - v) / (vMax - vMin)) * pH;

    // 零线 + token 刻度（轴名放刻度行右侧，省一行高度）
    ctx.strokeStyle = "#2a3454";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(padL + plotW, Y(0)); ctx.stroke();
    ctx.font = "18px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "#5d6a8f";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let n = 0; n < P; n += 20) { if (X(n) > padL + plotW - 170) break; ctx.fillText(String(n), X(n), pBot + 8); }
    ctx.textAlign = "right";
    ctx.fillText("token 下标 n →", padL + plotW, pBot + 8);

    const line = (arr, col, w) => {
      ctx.strokeStyle = col;
      ctx.lineWidth = w;
      ctx.beginPath();
      for (let n = 0; n < P; n++) { const px = X(n), py = Y(arr[n]); n ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke();
    };
    line(ideal, "rgba(224,183,120,.65)", 2);              // 教科书的全部 56 个频率
    // 理想曲线除 n = t 之外全是 0（与零线重合），所以把尖刺单独画粗一点，否则看不见
    ctx.fillStyle = "#e0b778";
    ctx.fillRect(X(o.t) - 2, Y(ideal[o.t]), 3.5, Y(0) - Y(ideal[o.t]));
    for (let j = 0; j < waves.length; j++) line(waves[j], PEAK_COLS[j % PEAK_COLS.length] + "99", 1.3);
    if (o.sumW) line(o.sumW, "rgba(143,227,242,.85)", 2.4);   // 按真实能量加权
    line(sum, o.col, 3.6);                                     // 等权叠加（粗线）

    // 峰值位置：虚线 + 白圈
    const xPk = X(o.t);
    ctx.strokeStyle = "rgba(255,255,255,.75)";
    ctx.setLineDash([7, 6]);
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xPk, pTop - 4); ctx.lineTo(xPk, pBot + 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(xPk, Y(sum[o.t]), 8, 0, Math.PI * 2);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 3;
    ctx.stroke();

    // 次高旁瓣（避开峰附近 ±3）
    let sb = -1, sv = -Infinity;
    for (let n = 0; n < P; n++) if (Math.abs(n - o.t) > 3 && sum[n] > sv) { sv = sum[n]; sb = n; }
    ctx.font = FONT_S;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillStyle = o.col;
    ctx.fillText(o.capLeft, padL, top + 4);
    ctx.fillStyle = "#7885a5";
    ctx.textAlign = "right";
    ctx.fillText(`次高旁瓣只有 ${sv.toFixed(3)}（n = ${sb}）→ 波峰唯一`, padL + plotW, top + 4);

    // 左侧槽位
    ctx.font = "16px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#8b97b5";
    ctx.fillText(o.sideTop, padL - 16, pTop + pH / 2 - 10);
    ctx.fillText(o.sideBottom, padL - 16, pTop + pH / 2 + 12);
  }
  const PEAK_COLS = ["#ff9f4a", "#5ad1a0", "#c77dff", "#ffd166", "#4cc9f0", "#9aa4ff", "#f78fb3", "#8fe3f2"];

  // ---------------- ③a 嵌入视图：理想 DFT 条带（128 维全画）+ 该向量的逆变换面板 ----------------
  function drawEmbedView() {
    const cv = $("#embedCanvas"), ctx = ctxOf(cv);
    const P = st.P, K = Math.floor(P / 2), D = st.d;   // P=113 → K=56；D=128
    clear(ctx, cv, "#0e1016");
    const padL = 132, padR = 40;
    const plotW = cv.width - padL - padR;
    const cw = plotW / D;                              // 每维一格
    const USED = 2 * K + 1;                            // 1 维直流 + 56×2 = 113 维有定义，余 15 维空闲

    // 头部说明
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.font = FONT_S;
    ctx.fillStyle = "#a3c4ff";
    ctx.fillText("理想化布局（教科书版）：一个频率 k 占两维 —— 第 1 维 = cos(θₖ)、第 2 维 = −sin(θₖ)，θₖ = 2πka/P", padL, 10);
    ctx.fillStyle = "#c9a15c";
    ctx.fillText(`完整的 ${D} 维：1 维直流 + ${K}×2 = ${2 * K} 维频率分量 + ${D - USED} 维空闲`, padL, 34);

    // ---- 布局：条带 a → 逆变换 a → 条带 b → 逆变换 b ----
    const rows = [
      { t: st.a, col: "#4ea1ff", label: "emb(a)", name: "a", on: !!st.sim.a && !!st.dimInfo },
      { t: st.b, col: "#ff7eb6", label: "emb(b)", name: "b", on: !!st.sim.b && !!st.dimInfo },
    ];
    const STRIP_H = 84, TICKS = 30, PANEL_H = 190, GAP = 32;
    let y = 96;
    for (const r of rows) {
      r.stripY = y; y += STRIP_H;
      r.tickY = y + 8; y += TICKS;
      if (r.on) { r.panelY = y; y += PANEL_H; }
      y += GAP;
    }
    const legendY = y - GAP + 10;
    const anyPanel = rows.some((r) => r.on);
    const needH = legendY + (anyPanel ? 106 : 56);
    if (cv.height !== needH) cv.height = needH;
    clear(ctx, cv, "#0e1016");

    const weights = st.dimInfo ? keyEnergyWeights() : [];

    for (const r of rows) {
      drawStripRow(r);
      if (r.on) drawInversePanel(r);
    }

    // ---- 底部图例 ----
    ctx.font = FONT_S;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#7885a5";
    ctx.fillText("条带：蓝 = 正值（向上）、粉 = 负值（向下），条高 = |取值|（1.0 满格）；右上角 🌀 图标 = 把该条带逆变换回 token 轴", padL, legendY);
    if (anyPanel) {
      const order = weights.map((w, j) => ({ w, k: st.keys[j] })).sort((a, b) => b.w - a.w);
      const top2 = order.slice(0, 2).map((t) => `k=${t.k} 占 ${(t.w * 100).toFixed(0)}%`).join("、");
      ctx.fillStyle = "#c9a15c";
      ctx.fillText(`逆变换用的频率 = 网络保留的 k = ${st.keys.join(", ")}（${top2}）`, padL, legendY + 26);
      ctx.fillStyle = "#7885a5";
      ctx.fillText("→ 频率少，峰就不尖：这正是第 5 步 logits 带毛刺的原因（所以判对判错只看同一输入下谁最高）", padL, legendY + 52);
    }

    // ---- 一条理想 DFT 条带：128 维全画 ----
    function drawStripRow(r) {
      // 区域标注（每条带子各自标一份：两个条带中间可能夹着逆变换面板）
      const regions = [
        { x0: padL, x1: padL + cw, text: "直流", col: "#7ce0a8" },
        { x0: padL + cw, x1: padL + USED * cw, text: `k = 1 … ${K}（每个频率两维：cos / −sin）`, col: "#4ea1ff" },
        { x0: padL + USED * cw, x1: padL + plotW, text: `空闲 ${D - USED} 维`, col: "#5d6a8f" },
      ];
      const ay = r.stripY - 12;
      for (const g of regions) {
        ctx.strokeStyle = g.col;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(g.x0 + 1, ay); ctx.lineTo(g.x1 - 1, ay); ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.font = "18px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = g.col;
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(g.text, (g.x0 + g.x1) / 2, ay - 4);
      }
      const midY = r.stripY + STRIP_H / 2;
      // 逐维取值：第 0 维是直流（恒为 1），第 2k−1 维 = cos、第 2k 维 = −sin
      for (let i = 0; i < USED; i++) {
        let v;
        if (i === 0) v = 1;
        else {
          const k = Math.ceil(i / 2);
          const th = (2 * Math.PI * k * r.t) / P;
          v = i % 2 === 1 ? Math.cos(th) : -Math.sin(th);
        }
        const av = Math.min(Math.abs(v), 1);
        const hh = (STRIP_H / 2) * av;
        ctx.fillStyle = v >= 0
          ? `rgba(78,161,255,${0.22 + av * 0.62})`
          : `rgba(255,126,182,${0.22 + av * 0.62})`;
        ctx.fillRect(padL + i * cw + 0.5, v >= 0 ? midY - hh : midY, Math.max(cw - 1, 1), Math.max(hh, 1));
      }
      // 空闲维底纹
      ctx.fillStyle = "#12151f";
      ctx.fillRect(padL + USED * cw, r.stripY, (D - USED) * cw, STRIP_H);
      ctx.strokeStyle = "#1e2433";
      ctx.lineWidth = 1;
      ctx.strokeRect(padL + USED * cw, r.stripY, (D - USED) * cw, STRIP_H);
      // 每两维一道分隔线（一个频率一组）
      ctx.strokeStyle = "#232a38";
      for (let k = 1; k <= K; k++) {
        const x = padL + 2 * k * cw;
        ctx.beginPath(); ctx.moveTo(x, r.stripY); ctx.lineTo(x, r.stripY + STRIP_H); ctx.stroke();
      }
      // 零线与行标
      ctx.strokeStyle = "#33405c";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(padL, midY); ctx.lineTo(padL + USED * cw, midY); ctx.stroke();
      ctx.font = FONT_B;
      ctx.fillStyle = r.col;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(r.label, padL - 16, midY);
      // 维度下标
      ctx.font = "18px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = "#5d6a8f";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i < D; i += 16) {
        if (padL + i * cw + cw / 2 > padL + plotW - 170) break;
        ctx.fillText(String(i), padL + i * cw + cw / 2, r.tickY);
      }
      ctx.textAlign = "right";
      ctx.fillText("维度下标 c →", padL + plotW, r.tickY);
    }

    // ---- 逆变换面板：把这条向量里的频率叠加回 token 轴，还原 one-hot 的唯一波峰 ----
    function drawInversePanel(r) {
      const K = st.keys.length;
      const waves = st.keys.map((k) => {
        const arr = new Float64Array(P);
        for (let n = 0; n < P; n++) arr[n] = Math.cos((2 * Math.PI * k * (r.t - n)) / P) / K;
        return arr;
      });
      const sum = new Float64Array(P);
      const sumW = new Float64Array(P);
      for (let n = 0; n < P; n++) {
        sum[n] = waves.reduce((s, w) => s + w[n], 0);
        sumW[n] = waves.reduce((s, w, j) => s + w[n] * weights[j] * K, 0);
      }
      drawPeakPanel(ctx, {
        top: r.panelY, padL, plotW, P, t: r.t, col: r.col, waves, sum, sumW,
        capLeft: `${r.label} 的逆变换：波峰在 n = ${r.t}（= one-hot 里的 ${r.name}）= ${sum[r.t].toFixed(3)}`,
        sideTop: "逆变换", sideBottom: "↕",
      });
    }
  }

  // ---------------- ③a′ 逆变换：把 emb 里的频率叠加起来 → 还原 one-hot 的唯一波峰 ----------------
  /** 每个关键频率在 W_E 里占的能量（归一化），逆变换面板的加权曲线用 */
  function keyEnergyWeights() {
    const w = st.keys.map((k, j) => st.dimInfo.reduce((s, t) => s + t.amps[j] ** 2, 0));
    const tot = w.reduce((a, b) => a + b, 0) || 1;
    return w.map((v) => v / tot);
  }

  // ---------------- ③b 真实权重：W_E[·, c] 的频谱结构 ----------------
  /** 右下那段说明文字随维度更新 */
  function updateEmbedNote() {
    const share = st.dimShare && st.dimShare.has(st.c) ? st.dimShare.get(st.c).toFixed(2) : "?";
    $("#realEmbedNote").innerHTML =
      `这一列 <code>W_E[·, c=${st.c}]</code> 的取值（浅蓝）与只用 k = ${st.keys.join(", ")} 重建的结果（橙）几乎重合——` +
      `说明这一列本质上就是 ${st.keys.length} 个 cos/sin 波的叠加，复现了 <b>${share}%</b>。`;
  }

  function drawRealEmbed() {
    const cv = $("#realEmbedCanvas"), ctx = ctxOf(cv);
    const P = st.P, d = st.d;
    if (!st.mats) return;
    clear(ctx, cv, "#0e1016");
    const WE = st.mats.token_embed.data;
    const y = new Float64Array(P);
    for (let a = 0; a < P; a++) y[a] = WE[a * d + st.c];
    const series = fourier(y, P);
    const keep = new Set(st.keys);
    const rec = rebuild(series, P, keep);
    const share = shareOf(y, P, series, keep);

    // —— 上半：真实取值 vs 只用关键频率的重建 ——
    const L = 110, R = 40, T = 60, plotW = cv.width - L - R;
    const midH = 190, midY = T + midH / 2;
    let vmax = 0;
    for (let n = 0; n < P; n++) vmax = Math.max(vmax, Math.abs(y[n]), Math.abs(rec[n]));
    vmax = vmax || 1;
    const X = (n) => L + (plotW * n) / (P - 1);
    const Y = (v) => midY - (midH / 2) * (v / vmax);

    ctx.strokeStyle = "#1a2033";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, midY); ctx.lineTo(L + plotW, midY); ctx.stroke();
    ctx.font = FONT_S;
    ctx.fillStyle = "#5d6a8f";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`+${vmax.toFixed(2)}`, L - 12, Y(vmax));
    ctx.fillText("0", L - 12, midY);
    ctx.fillText(`−${vmax.toFixed(2)}`, L - 12, Y(-vmax));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let n = 0; n < P; n += 20) ctx.fillText(String(n), X(n), Y(-vmax) + 10);

    const line = (arr, col, w) => {
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath();
      for (let n = 0; n < P; n++) { const px = X(n), py = Y(arr[n]); n ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke();
    };
    line(y, "#9db4cc", 1.6);          // 真实取值（浅蓝）
    line(rec, "#e07b1a", 3.4);        // 只用关键频率重建（橙）

    // 当前 a 的位置
    const ax = X(st.a);
    ctx.strokeStyle = "rgba(255,255,255,.85)";
    ctx.setLineDash([7, 6]);
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(ax, T - 6); ctx.lineTo(ax, T + midH + 8); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(ax, Y(rec[st.a]), 7, 0, Math.PI * 2); ctx.fill();
    ctx.font = FONT_B;
    ctx.fillStyle = "#8fe3f2";
    ctx.textAlign = ax > cv.width * 0.75 ? "right" : "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(`a = ${st.a} → ${rec[st.a].toFixed(3)}`, ax + (ax > cv.width * 0.75 ? -14 : 14), Y(rec[st.a]) - 12);

    ctx.font = FONT_S;
    ctx.fillStyle = "#a3c4ff";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(`真实权重：W_E[token, c=${st.c}] —— 128 维中的第 ${st.c} 维，沿 113 个 token 的取值（浅蓝）vs 只用 k = ${st.keys.join(", ")} 重建（橙）· 复现 ${share.toFixed(2)}%`, L, 16);

    // —— 下半：这一列的频谱，标出关键频率 ——
    const bT = T + midH + 62, bH = 150;
    const comps = series.comps;
    let amax = 0;
    for (const t of comps) amax = Math.max(amax, t.amp);
    amax = amax || 1;
    const bw = plotW / comps.length;
    for (let i = 0; i < comps.length; i++) {
      const t = comps[i];
      const x = L + i * bw;
      const hgt = (bH * t.amp) / amax;
      ctx.fillStyle = keep.has(t.k) ? "#e07b1a" : "rgba(120,140,180,.45)";
      ctx.fillRect(x, bT + bH - hgt, Math.max(bw - 1.5, 1), Math.max(hgt, 1));
    }
    ctx.font = FONT_S;
    ctx.fillStyle = "#7885a5";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(`这一列的频谱 |c_k|（横轴 k = 1…${comps.length}，橙 = 保留下来的关键频率）`, L, bT - 26);
    ctx.textAlign = "center";
    for (const k of st.keys) {
      const i = k - 1;
      if (i < 0 || i >= comps.length) continue;
      ctx.fillStyle = "#e0b878";
      ctx.fillText(String(k), L + i * bw + bw / 2, bT + bH + 8);
    }
  }

  // ---------------- ③c 真实权重：原样画出来的 128 维 + 维度×频率热图 + 真实逆变换 ----------------
  function drawRealDims() {
    const cv = $("#realDimsCanvas");
    if (!cv || !st.mats || !st.dimInfo) return;
    const ctx = ctxOf(cv);
    const D = st.d, K = st.keys.length, P = st.P;
    const WE = st.mats.token_embed.data;
    const padL = 150, padR = 40;
    const plotW = cv.width - padL - padR;
    const cw = plotW / D;
    const STRIP_H = 52, TICKS = 28, PANEL_H = 190, GAP = 30;

    // ---- 布局：条带 a → 逆变换 a → 条带 b → 逆变换 b → 维度×频率热图 → 结论 ----
    const rows = [
      { t: st.a, col: "#4ea1ff", label: "emb(a)", name: "a", on: !!st.sim2.a },
      { t: st.b, col: "#ff7eb6", label: "emb(b)", name: "b", on: !!st.sim2.b },
    ];
    let y = 60;
    for (const r of rows) {
      r.stripY = y; y += STRIP_H;
      r.tickY = y + 6; y += TICKS;
      if (r.on) { r.panelY = y; y += PANEL_H; }
      y += GAP;
    }
    const heatTitleY = y, hT = y + 26, rh = 20, hH = K * rh;
    const sumY = hT + hH + 12;
    const needH = sumY + 80;
    if (cv.height !== needH) cv.height = needH;
    clear(ctx, cv, "#0e1016");

    // ---- 头部 ----
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.font = FONT_S;
    ctx.fillStyle = "#8fe3f2";
    ctx.fillText("真实权重，原样画出来：这就是网络实际看到的 128 个数（没有任何理想化）", padL, 10);
    let rowMax = 0;
    for (const t of [st.a, st.b]) for (let c = 0; c < D; c++) rowMax = Math.max(rowMax, Math.abs(WE[t * D + c]));
    rowMax = rowMax || 1;
    ctx.fillStyle = "#c9a15c";
    ctx.fillText(`绝大多数维度都在 0 附近，只有少数几维是大值（满格 = ${rowMax.toFixed(2)}）—— 单看这个向量是看不出「加法算法」的`, padL, 34);

    for (const r of rows) {
      drawRealStrip(r);
      if (r.on) drawRealInverse(r);
    }
    drawHeat();
    drawSummary();

    // ---- 一条真实条带：W_E[t, :] 的 128 个真值 ----
    function drawRealStrip(r) {
      const mid = r.stripY + STRIP_H / 2;
      for (let c = 0; c < D; c++) {
        const v = WE[r.t * D + c];
        const av = Math.min(Math.abs(v) / rowMax, 1);
        const hh = (STRIP_H / 2) * av;
        ctx.fillStyle = v >= 0 ? `rgba(78,161,255,${0.22 + av * 0.66})` : `rgba(255,126,182,${0.22 + av * 0.66})`;
        ctx.fillRect(padL + c * cw + 0.5, v >= 0 ? mid - hh : mid, Math.max(cw - 1, 1), Math.max(hh, 1));
      }
      ctx.strokeStyle = "#33405c";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(padL, mid); ctx.lineTo(padL + plotW, mid); ctx.stroke();
      ctx.font = FONT_B;
      ctx.fillStyle = r.col;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(r.label, padL - 16, mid);
      ctx.font = "18px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.fillStyle = "#5d6a8f";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i < D; i += 16) {
        if (padL + i * cw + cw / 2 > padL + plotW - 170) break;
        ctx.fillText(String(i), padL + i * cw + cw / 2, r.tickY);
      }
      ctx.textAlign = "right";
      ctx.fillText("维度下标 c →", padL + plotW, r.tickY);
    }

    // ---- 真实逆变换：把 emb(t) 投影到这几个频率上，再叠加回 token 轴 ----
    function drawRealInverse(r) {
      const al = new Float64Array(K), be = new Float64Array(K);
      for (let j = 0; j < K; j++) {
        let A = 0, B = 0;
        for (let c = 0; c < D; c++) {
          const e = WE[r.t * D + c];
          A += e * st.dirA[j][c];      // ⟨emb(t), u_k⟩
          B += e * st.dirB[j][c];      // ⟨emb(t), v_k⟩
        }
        al[j] = A; be[j] = B;
      }
      const waves = st.keys.map((k, j) => {
        const arr = new Float64Array(P);
        for (let n = 0; n < P; n++) {
          arr[n] = al[j] * Math.cos((2 * Math.PI * k * n) / P) + be[j] * Math.sin((2 * Math.PI * k * n) / P);
        }
        return arr;
      });
      const sum = new Float64Array(P);
      for (let n = 0; n < P; n++) sum[n] = waves.reduce((s, w) => s + w[n], 0);
      // 归一化到峰高 1，方便与教科书的金线比尖度
      let amax = 0;
      for (let n = 0; n < P; n++) amax = Math.max(amax, Math.abs(sum[n]));
      amax = amax || 1;
      for (let n = 0; n < P; n++) {
        sum[n] /= amax;
        for (const w of waves) w[n] /= amax;
      }
      let pk = 0;
      for (let n = 0; n < P; n++) if (sum[n] > sum[pk]) pk = n;
      drawPeakPanel(ctx, {
        top: r.panelY, padL, plotW, P, t: r.t, col: r.col, waves, sum, normIdeal: true,
        capLeft: `${r.label} 用真实 W_E 的频率做逆变换：最大值在 n = ${pk}` +
          (pk === r.t ? `（= 真值 ${r.name} ✓）` : `（真值 ${r.name} 在 n = ${r.t}）`),
        sideTop: "逆变换", sideBottom: "↕",
      });
    }

    // ---- 热图：每个维度 × 6 个关键频率的幅度 ----
    function drawHeat() {
      ctx.font = FONT_S;
      ctx.fillStyle = "#a3c4ff";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`把同一块 W_E 拆到频域：每个维度在 ${K} 个关键频率上的幅度（横轴与上面两条带子对齐，都是维度下标 0…127）`, padL, heatTitleY);
      for (let j = 0; j < K; j++) {
        for (let c = 0; c < D; c++) {
          const a = Math.min(st.dimInfo[c].amps[j] / st.ampMax, 1);
          ctx.fillStyle = a > 0.004 ? `rgba(224,123,26,${0.1 + Math.sqrt(a) * 0.85})` : "#12151f";
          ctx.fillRect(padL + c * cw, hT + j * rh, Math.max(cw - 0.6, 1), rh - 1.5);
        }
        ctx.font = "18px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.fillStyle = "#e0b878";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(`k=${st.keys[j]}`, padL - 12, hT + j * rh + rh / 2);
      }
      ctx.strokeStyle = "#232a38";
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, hT, plotW, hH);
    }

    // ---- 文字总结：主频分组 + 整体复现 ----
    function drawSummary() {
      const grp = st.groups.map((g) => `k=${g.k}：${g.n} 维`).join(" · ");
      ctx.font = FONT_S;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#dfe3ea";
      ctx.fillText(`每维的“主频”分布（${grp}）——亮起来的那几列就是网络真正在用的维度`, padL, sumY);
      ctx.fillStyle = "#7ce0a8";
      ctx.fillText(`整块 W_E（128 维 × ${P} 个 token）只用这 ${K} 个频率就能复现 ${st.globalShare.toFixed(2)}% 的方差`, padL, sumY + 26);
      ctx.fillStyle = "#7885a5";
      ctx.fillText(`结构在频域里，不在原始维度顺序里 —— 所以上面那张理想化示意图必须“换个坐标基”才画得出来。`, padL, sumY + 52);
    }
  }

  // ---------------- ④ 旋转圆 ----------------
  function drawRotation() {
    const cv = $("#rotationCanvas"), ctx = ctxOf(cv);
    const P = st.P, W = cv.width, H = cv.height;
    clear(ctx, cv, "#0e1016");
    const cx = W / 2, cy = H / 2 - 6, R = 208;

    ctx.beginPath(); ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.strokeStyle = "#2a3454"; ctx.lineWidth = 3; ctx.stroke();
    ctx.strokeStyle = "#161b2c"; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - R - 40, cy); ctx.lineTo(cx + R + 40, cy);
    ctx.moveTo(cx, cy - R - 40); ctx.lineTo(cx, cy + R + 40);
    ctx.stroke();

    const thA = (2 * Math.PI * st.a) / P;
    const thB = (2 * Math.PI * st.b) / P;
    const thS = (2 * Math.PI * ((st.a + st.b) % P)) / P;
    const ray = (th, col, dashed) => {
      ctx.beginPath();
      if (dashed) ctx.setLineDash([10, 8]);
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + R * Math.cos(th), cy - R * Math.sin(th));
      ctx.strokeStyle = col; ctx.lineWidth = dashed ? 3 : 4; ctx.stroke();
      ctx.setLineDash([]);
    };
    ray(thA, "#4ea1ff"); ray(thB, "#ff7eb6"); ray(thS, "#6ee7a8", true);

    const dot = (th, col, label) => {
      const x = cx + R * Math.cos(th), y = cy - R * Math.sin(th);
      ctx.beginPath(); ctx.arc(x, y, 20, 0, Math.PI * 2);
      ctx.fillStyle = col; ctx.globalAlpha = 0.16; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2);
      ctx.fillStyle = col; ctx.fill();
      ctx.strokeStyle = "#0e1016"; ctx.lineWidth = 3; ctx.stroke();
      ctx.font = FONT_B; ctx.fillStyle = col;
      ctx.textAlign = Math.cos(th) >= 0 ? "left" : "right";
      ctx.textBaseline = "middle";
      ctx.fillText(label, x + (Math.cos(th) >= 0 ? 20 : -20), y - 16);
    };
    dot(thA, "#4ea1ff", `a = ${st.a}`);
    dot(thB, "#ff7eb6", `b = ${st.b}`);
    dot(thS, "#6ee7a8", `(a+b) mod ${P} = ${(st.a + st.b) % P}`);

    ctx.font = FONT_S;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillStyle = "#4ea1ff"; ctx.fillText("● a 转过的角度", 24, 20);
    ctx.fillStyle = "#ff7eb6"; ctx.fillText("● b 转过的角度", 24, 48);
    ctx.fillStyle = "#6ee7a8"; ctx.fillText("◌ 两者相加 = 答案的角度（这就是「频域加法」）", 24, 76);

    // 用一个关键频率举例：把「角度相加」标出来
    const k = st.keys[0] || 1;
    const deg = (n) => (360 * ((k * n) % P)) / P;
    ctx.font = FONT_S;
    ctx.fillStyle = "#7885a5";
    ctx.textAlign = "right";
    ctx.fillText(`取 k = ${k} 为例：a 转 ${deg(st.a).toFixed(1)}°、b 转 ${deg(st.b).toFixed(1)}°、`, W - 24, 20);
    ctx.fillText(`两者相加 ${(deg(st.a) + deg(st.b)).toFixed(1)}° ≡ ${deg(st.a + st.b).toFixed(1)}°（mod 360°）`, W - 24, 48);
  }

  // ---------------- ⑤ 打分：理想 vs 真实 ----------------
  function idealScores(a, b) {
    const P = st.P, K = Math.floor(P / 2);
    const out = new Float64Array(P);
    for (let c = 0; c < P; c++) {
      let s = 0;
      for (let k = 1; k <= K; k++) s += Math.cos((2 * Math.PI * k * (a + b - c)) / P);
      out[c] = s / K;
    }
    return out;
  }

  function drawScores() {
    const cv = $("#scoresCanvas"), ctx = ctxOf(cv);
    const P = st.P, W = cv.width;
    clear(ctx, cv, "#0e1016");
    const L = 110, R = 40, padW = W - L - R;
    const bw = padW / P;
    const ans = (st.a + st.b) % P;

    // ---- 上：理想化 ----
    const ideal = idealScores(st.a, st.b);
    const H1 = 150, T1 = 62;
    const Y1 = (v) => T1 + H1 * (1 - (v + 0.2) / 1.25);
    ctx.font = FONT_S;
    ctx.fillStyle = "#a3c4ff";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(`理想化：score(c) = Σ_{k=1…56} cos(2πk(a+b−c)/P) / 56`, L, 14);
    ctx.strokeStyle = "#2a3454"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(L, Y1(0)); ctx.lineTo(L + padW, Y1(0)); ctx.stroke();
    for (let c = 0; c < P; c++) {
      const y0 = Y1(0), y1 = Y1(ideal[c]);
      ctx.fillStyle = c === ans ? "#6ee7a8" : "rgba(78,161,255,.45)";
      ctx.fillRect(L + c * bw, Math.min(y0, y1), Math.max(bw - 0.8, 1), Math.max(Math.abs(y1 - y0), 1.2));
    }
    ctx.fillStyle = "#6ee7a8";
    ctx.font = FONT_B;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(`c = ${ans}：1.000`, L + ans * bw + bw / 2, T1 - 4);
    ctx.font = FONT_S;
    ctx.fillStyle = "#7885a5";
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText("其余 112 个候选都挤在这条线附近（|score| ≤ 0.11）", L + padW, Y1(0) - 6);

    // ---- 下：真实 logits ----
    const T2 = 250, H2 = 120;
    const lg = st.logits;
    if (lg) {
      let lo = Infinity, hi = -Infinity;
      for (let c = 0; c < P; c++) { lo = Math.min(lo, lg[c]); hi = Math.max(hi, lg[c]); }
      const span = hi - lo || 1;
      const Y2 = (v) => T2 + H2 * (1 - (v - lo) / span);
      ctx.font = FONT_S;
      ctx.fillStyle = "#8fe3f2";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`真实模型：这一对 (a, b) 现算出来的 113 个 logits（零线只是参考，看相对高低）`, L, T2 - 52);
      const zero = clamp(0, lo, hi);
      ctx.strokeStyle = "#2a3454"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L, Y2(zero)); ctx.lineTo(L + padW, Y2(zero)); ctx.stroke();
      const pred = lg.indexOf(hi);
      for (let c = 0; c < P; c++) {
        const y0 = Y2(zero), y1 = Y2(lg[c]);
        ctx.fillStyle = c === pred ? "#6ee7a8" : (c === ans ? "rgba(160,220,255,.75)" : "rgba(143,227,242,.34)");
        ctx.fillRect(L + c * bw, Math.min(y0, y1), Math.max(bw - 0.8, 1), Math.max(Math.abs(y1 - y0), 1.2));
      }
      ctx.fillStyle = "#6ee7a8";
      ctx.font = FONT_B;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(`ĉ = ${pred}`, L + pred * bw + bw / 2, T2 - 4);
    }

    // x 轴
    ctx.font = FONT_S;
    ctx.fillStyle = "#5d6a8f";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let c = 0; c < P; c += 10) ctx.fillText(String(c), L + c * bw + bw / 2, T2 + H2 + 12);
    ctx.fillText("候选答案 c", L + padW / 2, T2 + H2 + 40);

    // ---- 对照表 ----
    let iMax = -Infinity, i2 = -Infinity;
    for (let c = 0; c < P; c++) { if (ideal[c] > iMax) { i2 = iMax; iMax = ideal[c]; } else if (ideal[c] > i2) i2 = ideal[c]; }
    $("#idealPeak").textContent = iMax.toFixed(3);
    $("#idealOther").textContent = i2.toFixed(3);
    $("#idealGap").textContent = (iMax - i2).toFixed(3);

    if (lg) {
      const sorted = [...lg.keys()].sort((x, y) => lg[y] - lg[x]);
      const best = sorted[0], second = sorted[1];
      $("#realPeak").textContent = lg[best].toFixed(1);
      $("#realOther").textContent = `${second}（${lg[second].toFixed(1)}）`;
      $("#realGap").textContent = (lg[best] - lg[second]).toFixed(1);
      $("#realTop5").textContent = sorted.slice(0, 5).map((c) => `${c}:${lg[c].toFixed(0)}`).join(" · ");
      $("#realK").textContent = st.keys.length;
    }
  }

  // ---------------- ⑥ 结论 ----------------
  function drawFinal() {
    const P = st.P, ans = (st.a + st.b) % P;
    $("#finalAns").textContent = ans;
    if (!st.logits) return;
    const pred = st.logits.indexOf(Math.max(...st.logits));
    $("#finalPred").textContent = pred;
    $("#finalPredBox").className = "num-box " + (pred === ans ? "green" : "red");
    // 训练集 / 留出
    let where = "该 seed 没有 split.json";
    if (st.splitBytes) {
      const i = st.a * P + st.b;
      const isTrain = ((st.splitBytes[i >> 3] >> (i & 7)) & 1) === 1;
      where = isTrain
        ? `<b>训练集</b>（这一对在训练时见过：${st.splitMeta.n_train} / ${st.splitMeta.n} 对）`
        : `<b>留出集</b>（训练时没见过，用来检验泛化：${st.splitMeta.n_test} / ${st.splitMeta.n} 对）`;
    }
    $("#finalNote").innerHTML =
      `这一对 (a, b) = (${st.a}, ${st.b}) 属于 ${where}。` +
      `模型给 ${ans} 的分数是 ${st.logits[ans].toFixed(1)}，最高类是 ${pred}（${st.logits[pred].toFixed(1)}）——` +
      (pred === ans
        ? "预测正确。整条链没有「查表」，而是在频域里把 a、b 的相位相加、再读出最匹配的那个候选。"
        : "预测错了（早期权重或该 seed 还没泛化）。");
  }

  // ---------------- 载入权重 ----------------
  function pad6(n) { return String(n).padStart(6, "0"); }

  async function loadSeed(seed) {
    st.seed = seed;
    $("#loadChip").className = "rev-chip";
    $("#loadChip").textContent = `载入 seed ${seed} …`;
    try {
      const idx = await (await fetch("/viz-data/index.json", { cache: "no-cache" })).json();
      const run = Object.keys(idx.runs)[0];
      st.run = run;
      const sIdx = idx.runs[run].seeds[String(seed)];
      const cfg = sIdx.config;
      st.P = cfg.p; st.d = cfg.d_model; st.nh = cfg.n_heads; st.dh = cfg.d_head; st.nm = cfg.n_mlp;
      const epochs = (sIdx.epochs || []).slice().sort((x, y) => x - y);
      st.epoch = epochs[epochs.length - 1];
      const resp = await fetch(`/viz-data/${run}/seed${seed}/e${pad6(st.epoch)}.bin`, { cache: "no-cache" });
      if (!resp.ok) throw new Error(`快照 ${resp.status}`);
      const { mats } = parseBin(await resp.arrayBuffer());
      st.mats = mats;
      st.seedLabel = `seed ${seed} @ e${st.epoch}`;
      $("#loadChip").textContent = `权重已载入 · ${st.seedLabel}`;

      // 关键频率 + 每个维度的重建占比 + 每个维度的"主频"（真实条带/热图要用）
      st.keys = keyFreqsOf(mats.token_embed.data, st.P, st.d, 6);
      const keep = new Set(st.keys);
      const list = [];
      const info = [];
      const col = new Float64Array(st.P);
      let gSse = 0, gSst = 0;
      for (let c = 0; c < st.d; c++) {
        for (let a = 0; a < st.P; a++) col[a] = mats.token_embed.data[a * st.d + c];
        const s = fourier(col, st.P);
        const rec = rebuild(s, st.P, keep);
        let sse = 0, sst = 0;
        for (let n = 0; n < st.P; n++) { sse += (col[n] - rec[n]) ** 2; sst += (col[n] - s.c0) ** 2; }
        gSse += sse; gSst += sst;
        const sh = sst > 0 ? 100 * (1 - sse / sst) : 100;
        list.push({ c, share: sh });
        // 6 个关键频率上的幅度（按 st.keys 的顺序）与幅度最大的那个（"主频"）
        const amps = st.keys.map((k) => {
          const t = s.comps.find((q) => q.k === k);
          return t || { k, a: 0, b: 0, amp: 0 };
        });
        const dom = amps.slice().sort((x, y) => y.amp - x.amp)[0];
        info.push({
          c, amps: amps.map((t) => t.amp), dc: s.c0,
          cs: amps.map((t) => ({ k: t.k, a: t.a, b: t.b })),
          k: dom.k, R: dom.amp, phi: Math.atan2(dom.b, dom.a), share: sh,
        });
      }
      st.dimInfo = info;
      // 每个关键频率的「方向向量」：W_E[·, c] 在该频率上的 cos / sin 系数（长度 = d_model）
      st.dirA = st.keys.map((_, j) => Float64Array.from(info, (t) => t.cs[j].a));
      st.dirB = st.keys.map((_, j) => Float64Array.from(info, (t) => t.cs[j].b));
      st.globalShare = gSst > 0 ? 100 * (1 - gSse / gSst) : 100;   // 整块 W_E 用 6 个频率复现的方差占比
      st.ampMax = Math.max(...info.flatMap((t) => t.amps), 1e-9);
      // 每个频段里有多少维把它当"主频"
      const groups = st.keys.map((k) => ({ k, n: info.filter((t) => t.k === k).length }));
      groups.sort((x, y) => y.n - x.n);
      st.groups = groups.filter((g) => g.n > 0);
      list.sort((x, y) => y.share - x.share);
      st.dims = list;
      st.dimShare = new Map(list.map((t) => [t.c, t.share]));
      if (!list.some((t) => t.c === st.c)) st.c = list[0].c;
      const sel = $("#dimSel");
      sel.innerHTML = "";
      for (const t of list.slice(0, 40)) sel.add(new Option(`c=${t.c} · 复现 ${t.share.toFixed(1)}%`, String(t.c)));
      sel.value = String(st.c);
      // 右上角两组图标现在可用（它们要用刚算好的频率系数）
      for (const id of ["#simA", "#simB", "#simA2", "#simB2"]) $(id).disabled = false;

      // 关键频率相关的文字
      $("#kCount").textContent = String(st.keys.length);
      $("#kList").textContent = st.keys.join(", ");
      const best = list[0];
      $("#weShare").textContent = `${best.share.toFixed(2)}%（全网络复现最好的维度是 c=${best.c}）`;
      updateEmbedNote();
      $("#realRotNote").textContent = `k = ${st.keys.join(", ")}，其余频率上权重≈0`;

      // split.json（训练/留出）
      try {
        const sp = await (await fetch(`/viz-data/${run}/seed${seed}/split.json`, { cache: "no-cache" })).json();
        const bin = atob(sp.train);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        st.splitBytes = bytes;
        st.splitMeta = sp;
      } catch (e) { st.splitBytes = null; }

      updateAll();
      return true;
    } catch (err) {
      console.error(err);
      $("#loadChip").className = "rev-chip err";
      $("#loadChip").textContent = `载入失败：${err.message}`;
      return false;
    }
  }

  // ---------------- 主更新 ----------------
  function updateAll() {
    if (st.mats) {
      st.logits = forward(st.a, st.b);
      const P = st.P, i = st.a * P + st.b;
      let tag = "—";
      if (st.splitBytes) {
        const isTrain = ((st.splitBytes[i >> 3] >> (i & 7)) & 1) === 1;
        tag = isTrain ? `(a,b) 属于训练集` : `(a,b) 属于留出集（测试）`;
      }
      $("#splitChip").textContent = tag;
    }
    drawNumbers();
    drawOneHot();
    drawEmbedView();
    drawRealDims();
    drawRealEmbed();
    drawRotation();
    drawScores();
    drawFinal();
  }

  async function init() {
    // 滑块
    const aS = $("#aSlider"), bS = $("#bSlider");
    aS.max = bS.max = String(st.P - 1);
    // 浏览器可能恢复表单状态（或自动填充），这里以脚本状态为准，保证每次打开都是 (15, 27)
    aS.value = String(st.a);
    bS.value = String(st.b);
    aS.addEventListener("input", () => { st.a = Number(aS.value); updateAll(); });
    bS.addEventListener("input", () => { st.b = Number(bS.value); updateAll(); });
    const ss = $("#seedSel");
    try {
      const idx = await (await fetch("/viz-data/index.json", { cache: "no-cache" })).json();
      const run = Object.keys(idx.runs)[0];
      const seeds = Object.keys(idx.runs[run].seeds).sort((x, y) => Number(x) - Number(y));
      seeds.forEach((s) => ss.add(new Option(`seed ${s}`, s)));
      ss.value = seeds[0];
    } catch (e) {
      ss.add(new Option("（没有 viz-data）", "0"));
    }
    ss.addEventListener("change", () => loadSeed(ss.value));
    $("#dimSel").addEventListener("change", (e) => { st.c = Number(e.target.value); updateEmbedNote(); drawRealEmbed(); });
    // 画布右上角：用频率做逆变换（③a 用理想频率、③c 用真实 W_E 的频率）
    const wireSim = (btn, key, simState, redraw) => {
      $(btn).addEventListener("click", () => {
        if (!st.dimInfo) return;
        simState[key] = !simState[key];
        $(btn).classList.toggle("on", simState[key]);
        redraw();
      });
    };
    wireSim("#simA", "a", st.sim, drawEmbedView);
    wireSim("#simB", "b", st.sim, drawEmbedView);
    wireSim("#simA2", "a", st.sim2, drawRealDims);
    wireSim("#simB2", "b", st.sim2, drawRealDims);
    await loadSeed(ss.value);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
