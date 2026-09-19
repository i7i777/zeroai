/* 模型可视化 —— 网络结构 + 按 seed/阶段加载权重做研究
 * 数据来源: python/grokking/vizexport.py 导出的 /viz-data/**（.bin 快照 + index.json）
 */
(() => {
  "use strict";

  // ---------------- 小工具 ----------------
  const $ = (s, el) => (el || document).querySelector(s);
  const pad6 = (n) => String(n).padStart(6, "0");

  // ---------------- 快照解析 ----------------
  /** 取 JSON；默认带 no-cache 条件请求，避免 split.json 这类会被重新生成的文件取到旧副本 */
  async function fetchJSON(u, opts) {
    const r = await fetch(u, opts);
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
    snapCache: new Map(),     // (run|seed|epoch) → mats（多 seed 参数对比用，带缓存）
    splitCache: new Map(),    // (run|seed) → 训练/测试划分位图（模加法表用）
    addGen: 0,                // 模加法表的重绘代号，用来丢弃过期的异步结果（切 seed 时会变）
    seedMode: "curve",        // ⓪ 卡片的视角：curve = 效果，param = 参数（W_E 频率谱）
    paramGen: 0,              // 参数视图的“代”号，用来丢弃过期的异步结果
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
    freqSrc: null,     // 当前打开的频率视图：embed/attn/mlp/logits
    freqPick: {},      // 每级自动挑中的“最正弦”维度/类别（换权重后清空）
    archStage: null,   // 结构图上高亮的级（与按钮/面板同步）
  };

  // ===================================================================
  // 网络结构 SVG（类似论文/示意图风格）
  // ===================================================================
  /** 还没跑过测试时，结构图里默认展示的那一对输入 (a, b)
   *  （EMBED 视图的默认 token 也要用它，两处必须一致） */
  function defaultSampleAB(p) {
    return { a: Math.max(1, Math.floor(p * 0.33)), b: Math.max(1, Math.floor(p * 0.62)) };
  }

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
    const wIn = 210, wEmb = 140, wAtt = 270, wMlp = 330, wLog = 76;
    const inX = X0;
    const embX = inX + wIn + GAP;
    const attX = embX + wEmb + GAP;
    const mlpX = attX + wAtt + GAP;
    const logX = mlpX + wMlp + GAP;        // MLP 之后就是读出的 logits（UNEMBED 块已删）
    const inCx = inX + wIn / 2, embCx = embX + wEmb / 2, attCx = attX + wAtt / 2,
          mlpCx = mlpX + wMlp / 2, logCx = logX + wLog / 2;
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

    // —— 输入 / EMBED / ATTENTION / MLP 整体下移（垂直居中的“流水线”上半区）——
    s += `<g transform="translate(0,${VS})">`;

    // 箭头（每段等长 GAP）
    const arrows = [[inX + wIn, embX], [embX + wEmb, attX], [attX + wAtt, mlpX], [mlpX + wMlp, logX]];
    for (const [x1, x2] of arrows) {
      s += `<line x1="${x1}" y1="${AY}" x2="${x2}" y2="${AY}" stroke="#8b2f2f" stroke-width="2.4" marker-end="url(#arr)"/>`;
    }

    // 1) 输入：3 个 one-hot token（位置 a、b、=），并标出 a、b 代表的数值
    const NTOK = p + 1;                         // 词表：0..112 与 '='
    const hasS = !!sample;
    const dflt = defaultSampleAB(p);
    const aVal = hasS ? sample.a : dflt.a;
    const bVal = hasS ? sample.b : dflt.b;
    const tokVals = [aVal, bVal, p];            // 第三列为 '=' token
    const tokTags = [
      hasS ? `位置1 · a=${aVal}  ●测试` : `位置1 · a=${aVal}`,
      hasS ? `位置2 · b=${bVal}  ●测试` : `位置2 · b=${bVal}`,
      `位置3 · '='`,
    ];
    // —— 真实前向：四个方面块里画的是“当前样本 × 当前权重”算出来的输出张量 ——
    const T = archTensors(aVal, bVal);
    const embVmax = T ? vmaxOf(T.emb) : 1;                              // 条带缩放基准（各块各自归一）
    const attVmax = T ? vmaxOf([...T.heads, T.attn]) : 1;
    const logMax = T ? Math.max(...T.logits) : 1;
    const logMin = T ? Math.min(...T.logits) : 0;
    const predIdx = T ? T.logits.indexOf(logMax) : -1;
    /** 权重还没载入时块内的占位 */
    const pending = (x, y, w, hh) =>
      `<rect x="${x}" y="${y}" width="${w}" height="${hh}" rx="5" fill="#1b2130" stroke="#2c3446"/>` +
      `<text x="${x + w / 2}" y="${y + hh / 2 + 4}" fill="#7f8ba3" font-size="10" text-anchor="middle" font-family="sans-serif">权重载入中…</text>`;

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

    // 2) EMBED：框里画的是「3 个 token 各自查表 + 位置编码」以后的输出张量（3×d）
    //    每个向量单独成组（data-vec），便于高亮到「某一个 token 的嵌入向量」而不是整块 EMBED
    s += `<g data-stage="embed"><rect x="${embX}" y="150" width="${wEmb}" height="320" rx="10" fill="url(#bx)" stroke="#c0843c"/>`;
    s += lbl(embCx, 140, "EMBED", 13, "#c9a15c");
    s += lbl(embCx, 182, `输出张量 3×${d}`, 10, "#9aa3b2");
    s += lbl(embCx, 196, `= W_E[token] + pos`, 9, "#7f8ba3");
    const embVecs = [
      { key: "a", val: tokVals[0], col: "#ffd57a", tag: "位置1" },   // 位置1 的 token a
      { key: "b", val: tokVals[1], col: "#9db4d0", tag: "位置2" },   // 位置2 的 token b
      { key: "=", val: tokVals[2], col: "#9db4d0", tag: "位置3" },   // 位置3 的 '='
    ];
    const vX = embX + 16, vW = wEmb - 32, vH = 62, vTop = 208, vGap = 20;
    for (let i = 0; i < embVecs.length; i++) {
      const v = embVecs[i], vy = vTop + i * (vH + vGap);
      s += `<g data-vec="${v.key}">`;
      s += `<rect x="${vX}" y="${vy}" width="${vW}" height="${vH}" rx="7" fill="#2a3145" stroke="#6b7385"/>`;
      s += T ? strip(T.emb[i], vX + 8, vy + 8, vW - 16, 40, embVmax)
             : pending(vX + 8, vy + 8, vW - 16, 40);
      // 两行文字：名字左对齐、位置右对齐，避免同一行挤在一起
      s += `<text class="vecName" x="${vX + 8}" y="${vy + vH - 9}" fill="${v.col}" font-size="9.5" font-family="monospace">W_E[${v.val}] + p${i + 1}</text>`;
      s += `<text x="${vX + vW - 8}" y="${vy + vH - 9}" fill="#9aa3b2" font-size="9" text-anchor="end" font-family="monospace">${d} 维</text>`;
      s += `</g>`;
    }
    s += lbl(embCx, 452, `条带 = ${d} 个分量`, 10.5);
    s += `</g>`;

    // 3) ATTENTION（4 头）
    //    框里画的是每个头「softmax(QKᵀ/√d)·V → W_O」以后的输出张量（各 d 维），
    //    以及 4 个头相加的结果 —— 那才是频率视图分析的那一层（位置「=」的残差流增量）。
    s += `<g data-stage="attn"><rect x="${attX}" y="70" width="${wAtt}" height="480" rx="12" fill="#1f2530" stroke="#7a5bb0"/>`;
    s += lbl(attCx, 55, `ATTENTION（${h} 头）`, 14, "#b18be0");
    const L = attX + 22;
    const HW = wAtt - 44;
    const headH = 92, top0 = 88, gapV = 8;
    const rail = L + HW + 10;                 // 右侧汇集线：把 4 个头的输出收拢到 Σ 框
    const sumY = 486, sumH = 50;
    const taps = [];
    for (let i = 0; i < h; i++) {
      const yy = top0 + i * (headH + gapV);
      s += `<rect x="${L}" y="${yy}" width="${HW}" height="${headH}" rx="8" fill="#2a3040" stroke="#5b6172"/>`;
      s += `<text x="${L + 12}" y="${yy + 16}" fill="#9aa3b2" font-size="10">head ${i}</text>`;
      s += `<text x="${L + HW - 12}" y="${yy + 16}" fill="#7f8ba3" font-size="9" text-anchor="end">W_O·(attn·V) 输出 · ${d} 维</text>`;
      s += T ? strip(T.heads[i], L + 12, yy + 24, HW - 24, 54, attVmax)
             : pending(L + 12, yy + 24, HW - 24, 54);
      taps.push(yy + 24 + 27);                // 该头输出条带的纵向中心
    }
    // 汇集线：每个头的输出 → 右侧竖线 → 底部 Σ 框
    for (const ty of taps) {
      s += `<line x1="${L + HW}" y1="${ty}" x2="${rail}" y2="${ty}" stroke="#c3a1ee" stroke-width="1.4"/>`;
    }
    s += `<line x1="${rail}" y1="${taps[0]}" x2="${rail}" y2="${sumY + sumH / 2}" stroke="#c3a1ee" stroke-width="1.4"/>`;
    s += `<line x1="${rail}" y1="${sumY + sumH / 2}" x2="${L + HW}" y2="${sumY + sumH / 2}" stroke="#c3a1ee" stroke-width="1.6" marker-end="url(#arrL)"/>`;
    // Σ 子层：频率视图（🎯 ATTENTION 频率）分析的就是这一层
    s += `<g data-part="attnSum">`;
    s += `<rect x="${L}" y="${sumY}" width="${HW}" height="${sumH}" rx="8" fill="#2a3040" stroke="#8a90a3"/>`;
    s += `<text class="partName" x="${L + 12}" y="${sumY + 15}" fill="#e0c25a" font-size="10.5">Σ 4 头输出相加 → W_O 投影（${d} 维，位置「=」）</text>`;
    s += T ? strip(T.attn, L + 12, sumY + 24, HW - 24, 22, attVmax)
           : pending(L + 12, sumY + 26, HW - 24, 20);
    s += `</g>`;
    s += lbl(attCx, 580, "每个头：softmax(QKᵀ/√d_head)·V（d_head=" + (cfg ? cfg.d_head : 32) + "）；条带 = 该头经 W_O 之后的输出张量", 10.5);
    s += `</g>`;          // 关闭 ATTENTION 分组
    s += `</g>`;          // 关闭 translate（MLP 用绝对坐标、占满全高，下面单独画）

    // 4) MLP —— 框里画的是三层各自的输出张量（128 输入 → 512 ReLU 隐层 → 128 输出）
    //    boxY=70 / boxH=480：与上方 ATTENTION 完全等高、同顶（70→550）
    {
      const boxY = 70, boxH = 480;
      s += `<g data-stage="mlp"><rect x="${mlpX}" y="${boxY}" width="${wMlp}" height="${boxH}" rx="12" fill="url(#bx)" stroke="#2e8b66"/>`;
      s += lbl(mlpCx, boxY - 15, "MLP · 单隐层全连接", 13, "#57c08a");   // 标题在框上方，与 ATTENTION 标题同线
      s += lbl(mlpCx, boxY + 18, `${d} → ${m} → ${d}（+ReLU）· 各层输出张量`, 10.5);
      const sx = mlpX + 18, sw = wMlp - 36;

      // ① 输入：残差流（位置「=」）；三条条带各自归一，标签里给出实际峰值
      const pIn = T ? peakOf(T.resid2) : 0, pHid = T ? peakOf(T.hid) : 0, pOut = T ? peakOf(T.mlpOut) : 0;
      const pk = (v) => `max|·| = ${v.toFixed(2)}`;
      s += `<g data-part="mlpIn">`;
      s += `<rect x="${sx}" y="126" width="${sw}" height="66" rx="6" fill="none" stroke="#3d4a63" stroke-dasharray="5 5"/>`;
      s += `<text class="partName" x="${sx + 9}" y="140" fill="#8fd0ff" font-size="10">输入 resid2（${d} 维 · 位置「=」）</text>`;
      if (T) s += `<text x="${sx + sw - 9}" y="140" fill="#7f8ba3" font-size="9" text-anchor="end" font-family="monospace">${pk(pIn)}</text>`;
      s += T ? strip(T.resid2, sx + 10, 148, sw - 20, 36, pIn || 1) : pending(sx + 10, 148, sw - 20, 36);
      s += `</g>`;

      // ② 隐层：ReLU 之后（512 维，偏密 → 连续竖条）
      s += `<rect x="${sx}" y="222" width="${sw}" height="150" rx="6" fill="none" stroke="#3d4a63" stroke-dasharray="5 5"/>`;
      s += `<text x="${sx + 9}" y="236" fill="#8ae3a8" font-size="10">隐层 ReLU 激活（${m} 维）</text>`;
      if (T) s += `<text x="${sx + sw - 9}" y="236" fill="#7f8ba3" font-size="9" text-anchor="end" font-family="monospace">${pk(pHid)}</text>`;
      s += T ? strip(T.hid, sx + 10, 244, sw - 20, 118, pHid || 1) : pending(sx + 10, 244, sw - 20, 118);

      // ③ 输出：W_out·ReLU(…) + b_out —— 加回残差流的 128 维增量
      //    「MLP 频率」分析的就是这一层，所以点亮的是它
      s += `<g data-part="mlpOut">`;
      s += `<rect x="${sx}" y="392" width="${sw}" height="66" rx="6" fill="none" stroke="#3d4a63" stroke-dasharray="5 5"/>`;
      s += `<text class="partName" x="${sx + 9}" y="406" fill="#c9a15c" font-size="10">输出 = W_out·ReLU(…) + b_out（${d} 维）</text>`;
      if (T) s += `<text x="${sx + sw - 9}" y="406" fill="#7f8ba3" font-size="9" text-anchor="end" font-family="monospace">${pk(pOut)}</text>`;
      s += T ? strip(T.mlpOut, sx + 10, 414, sw - 20, 36, pOut || 1) : pending(sx + 10, 414, sw - 20, 36);
      s += `</g>`;

      s += lbl(mlpCx, boxY + boxH - 30, "每层条带各自归一（幅度对比看右侧 max|·|）；琥珀 = 正，天蓝 = 负", 8.5);
      s += lbl(mlpCx, boxY + boxH - 14, `尺寸按 PyTorch 存法 = [输出维 × 输入维]：W_in ${m}×${d}，W_out ${d}×${m}`, 8.5);
      s += `</g>`;
    }
    // 重新平移：画读出端
    s += `<g transform="translate(0,${VS})">`;
    // 5) 读出端：直接把 113 个类的 logit 画出来（未单独画 UNEMBED 权重块：
    //    这块权重的作用就是把残差流读出成 logits，已在 📊 LOGITS 频率面板里体现）
    const RD_TOP = 150, RD_H = 312, RD_ROW = RD_H / p;
    const rdClass = (hasS && sample.c != null) ? sample.c : (aVal + bVal) % p;   // 真值类（金 = 真值）
    s += `<g data-stage="logits"><rect x="${logX}" y="120" width="${wLog}" height="380" rx="8" fill="url(#bx)" stroke="#c0843c"/>`;
    s += lbl(logCx, 100, "logits", 11, "#c9a15c");
    s += lbl(logCx, 138, T ? `W_U·x → ${p} 类` : `W_U·x → ${p} 类`, 8, "#9aa3b2");
    {
      const gx = logX + 10, gw = wLog - 40;
      const span = Math.max(1e-9, logMax - logMin);
      for (let r = 0; r < p; r++) {                 // 一格 = 一个输出类的 logit（条长 ∝ 数值大小）
        const cy = (RD_TOP + r * RD_ROW).toFixed(1);
        const rh = Math.max(1, RD_ROW - 0.6).toFixed(2);
        s += `<g data-part="logitsCell" data-idx="${r}">`;
        s += `<rect x="${gx}" y="${cy}" width="${gw}" height="${rh}" fill="#273149"/>`;
        if (T) {
          const rel = (T.logits[r] - logMin) / span;            // 0…1，本样本自己的相对大小
          const col = r === predIdx ? "#7ce0a8" : (rel > 0.72 ? "#8fe3f2" : rel > 0.42 ? "#4f9cbf" : "#33607a");
          s += `<rect x="${gx}" y="${cy}" width="${Math.max(0.8, rel * gw).toFixed(2)}" height="${rh}" fill="${col}"/>`;
        }
        s += `<text x="${gx + gw + 5}" y="${(RD_TOP + r * RD_ROW + RD_ROW - 0.4).toFixed(1)}" fill="#8fe3f2" font-size="8" font-family="monospace">${r}</text>`;
        s += `</g>`;
      }
      const tY = RD_TOP + rdClass * RD_ROW;         // 真值类在第几格
      s += `<rect x="${gx - 3}" y="${(tY - 1).toFixed(1)}" width="${gw + 6}" height="${(RD_ROW + 2).toFixed(1)}" rx="1.5" fill="none" stroke="#ffd57a" stroke-width="0.8"/>`;
      s += lbl(logCx, 484, "条长 ∝ logit（金框 = 真值类）", 7.5, "#9aa3b2");
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

  // ===================================================================
  // 模加法表：P×P 个输入对 (a, b) 的训练 / 测试划分
  // ===================================================================
  const ADD_CELL = 8;                         // 每格的画布像素（113×8 = 904，显示时缩到 400px）
  const COL_TRAIN = "#4d8fd6", COL_TEST = "#ff7a6b";

  /** 取该 (run,seed) 的完整训练掩码（split.json：位图 + base64） */
  async function loadSplit() {
    const key = `${state.run}#${state.seed}`;
    if (state.splitCache.has(key)) return state.splitCache.get(key);
    let sp = null;
    try {
      const d = await fetchJSON(`/viz-data/${state.run}/seed${state.seed}/split.json`,
                                { cache: "no-cache" });
      const bin = atob(d.train);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      sp = { p: d.p, n: d.n, nTrain: d.n_train, nTest: d.n_test, frac: d.frac_train, bytes };
    } catch (e) {
      console.warn("split.json 取不到（跑 python/_gen_testpairs.py 生成）：", e);
    }
    state.splitCache.set(key, sp);
    return sp;
  }

  /** 第 (a,b) 对是训练集吗（位序：第 j 字节的第 i 位 = mask[j*8 + i]） */
  function isTrainCell(sp, a, b) {
    const i = a * sp.p + b;
    return ((sp.bytes[i >> 3] >> (i & 7)) & 1) === 1;
  }

  async function drawAddTable() {
    const cv = $("#cAdd");
    if (!cv || !state.run || state.seed == null) return;
    const gen = ++state.addGen;
    const sp = await loadSplit();
    if (gen !== state.addGen) return;                     // 期间又切了 seed，丢弃这次结果
    const hint = $("#addHint"), legend = $("#addLegend"), tip = $("#addTip");
    if (tip) tip.hidden = true;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#11141d";
    ctx.fillRect(0, 0, cv.width, cv.height);
    if (!sp) {
      cv.hidden = true;
      if (hint) hint.textContent = "没有 split.json";
      if (legend) legend.textContent = "缺 split.json：跑一次 python/_gen_testpairs.py 即可补上（只算数据划分，不用重训）";
      return;
    }
    cv.hidden = false;
    const p = sp.p, cell = ADD_CELL;
    if (cv.width !== p * cell) { cv.width = cv.height = p * cell; }

    // 坐标轴刻度：写在 HTML 里（画布缩到 400px 后 canvas 里的 6px 小字看不清）
    const axX = $("#addAxisX"), axY = $("#addAxisY");
    if (axX && axY) {
      if (axX.dataset.p !== String(p)) {
        const pos = (k) => `${(((k + 0.5) / p) * 100).toFixed(3)}%`;
        let spanX = "", spanY = "";
        for (let k = 0; k < p; k += 10) {
          spanX += `<span style="left:${pos(k)}">${k}</span>`;
          spanY += `<span style="top:${pos(k)}">${k}</span>`;
        }
        axX.innerHTML = spanX;
        axY.innerHTML = spanY;
        axX.dataset.p = axY.dataset.p = String(p);
      }
    }

    // 每格一对输入：蓝 = 训练集，红 = 留出（测试）
    for (let a = 0; a < p; a++) {
      const y = a * cell;
      for (let b = 0; b < p; b++) {
        ctx.fillStyle = isTrainCell(sp, a, b) ? COL_TRAIN : COL_TEST;
        ctx.fillRect(b * cell, y, cell - 1, cell - 1);
      }
    }

    // 当前测试样本那一格加白描边
    const s = state.sample;
    if (s && s.a < p && s.b < p) {
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
      ctx.strokeRect(s.b * cell - 1, s.a * cell - 1, cell + 1, cell + 1);
    }

    const pc = (n) => `${((100 * n) / sp.n).toFixed(1)}%`;
    if (hint) hint.textContent = `${p}×${p} = ${sp.n} 对 · 横轴 b / 纵轴 a · seed ${state.seed}`;
    if (legend) {
      legend.innerHTML =
        `<span><b>训练集</b> ${sp.nTrain} 对（${pc(sp.nTrain)}）</span>` +
        `<span><b>留出（测试）</b> ${sp.nTest} 对（${pc(sp.nTest)}）</span>` +
        `<span>P = ${p} · frac_train = ${sp.frac}</span>`;
    }
  }

  /** 鼠标悬停显示这一对 (a,b) 的真值与归属 */
  function addTableTip(ev) {
    const cv = $("#cAdd"), tip = $("#addTip");
    if (!cv || !tip || cv.hidden) return;
    const sp = state.splitCache.get(`${state.run}#${state.seed}`);
    if (!sp) return;
    const r = cv.getBoundingClientRect();
    const b = Math.floor(((ev.clientX - r.left) * (cv.width / r.width)) / ADD_CELL);
    const a = Math.floor(((ev.clientY - r.top) * (cv.height / r.height)) / ADD_CELL);
    if (a < 0 || b < 0 || a >= sp.p || b >= sp.p) { tip.hidden = true; return; }
    const wr = cv.parentElement.parentElement.getBoundingClientRect();
    tip.hidden = false;
    tip.style.left = `${ev.clientX - wr.left}px`;
    tip.style.top = `${ev.clientY - wr.top}px`;
    tip.innerHTML = `a = <b>${a}</b>, b = <b>${b}</b> → c = <b>${(a + b) % sp.p}</b> · ` +
                    (isTrainCell(sp, a, b) ? "训练集" : "留出（测试）");
  }

  // ===================================================================
  // 卡片全屏：右上角 ⛶ 点一下把整张卡片铺满屏幕（Fullscreen API，退化用 class）
  // ===================================================================
  /** 卡片是否处于全屏态 */
  function cardIsFull(name) {
    const el = document.querySelector(`[data-fs="${name}"]`);
    if (!el) return false;
    return document.fullscreenElement === el || el.classList.contains("fs-fallback");
  }

  /** 各卡片进/退全屏后的重画钩子（以后想让别的卡片也能全屏，在这里补一条） */
  const FS_REDRAW = { acc: () => drawAcc() };

  /** 图标跟着状态变：全屏时显示 ✕ */
  function syncFsBtns() {
    for (const btn of document.querySelectorAll("[data-fs-btn]")) {
      const card = btn.closest("[data-fs]");
      const on = card ? cardIsFull(card.dataset.fs) : false;
      btn.textContent = on ? "✕" : "⛶";
      btn.title = on ? "退出全屏（Esc）" : "全屏";
      btn.setAttribute("aria-label", on ? "退出全屏" : "全屏");
    }
  }

  async function toggleCardFullscreen(card) {
    const name = card.dataset.fs;
    const active = document.fullscreenElement === card || card.classList.contains("fs-fallback");
    if (active) {
      if (document.fullscreenElement === card) {
        try { await document.exitFullscreen(); } catch (err) { /* 忽略：状态由 fullscreenchange 收尾 */ }
      }
      card.classList.remove("fs-fallback");
    } else {
      let ok = false;
      if (card.requestFullscreen) ok = await card.requestFullscreen().then(() => true).catch(() => false);
      if (!ok) card.classList.add("fs-fallback");       // 不支持 / 被拒绝时的退化方案
    }
    syncFsBtns();
    if (FS_REDRAW[name]) FS_REDRAW[name]();
  }

  // -------------------------------------------------------------------
  // 选中 seed 的准确率曲线（训练 / 测试），标出顿悟点与当前 epoch
  // -------------------------------------------------------------------
  function drawAcc() {
    const cv = $("#cAcc");
    if (!cv) return;
    // 全屏时换一套更大的设计尺寸：量出卡片里除画布外的占用，让画布吃掉剩下的空间
    const full = cardIsFull("acc");
    let W = 1100, H = 260, dpr = 1;
    if (full) {
      const card = $("#accCard");
      const cs = getComputedStyle(card);
      const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
      let others = 0;                                     // 标题 / 图例 / 说明 + 各自的外边距
      for (const el of card.children) {
        if (el.tagName === "CANVAS" || el.matches("[data-fs-btn]")) continue;
        const r = el.getBoundingClientRect();
        if (!r.height) continue;
        const s = getComputedStyle(el);
        others += r.height + parseFloat(s.marginTop) + parseFloat(s.marginBottom);
      }
      W = Math.max(880, Math.round(cv.clientWidth || card.clientWidth));
      H = Math.max(300, Math.round(card.clientHeight - padY - others));
      dpr = Math.min(2, Math.max(1.5, window.devicePixelRatio || 1));
    }
    // 缓冲区按设备像素比放大，避免全屏时被拉糊；绘图坐标仍用设计尺寸
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const L = 62, R = 26, T = 18, B = 34;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#11141d"; ctx.fillRect(0, 0, W, H);

    const legend = $("#accLegend");
    const curve = state.idx?.curve;
    const ep = curve?.epoch, tr = curve?.train_acc, te = curve?.test_acc;
    if (!ep || !te) {
      ctx.fillStyle = "#9aa3b2"; ctx.font = "13px sans-serif";
      ctx.fillText("没有准确率曲线数据（viz-data/index.json 里缺少 curve）", L, H / 2);
      if (legend) legend.textContent = "—";
      return;
    }
    const e0 = ep[0], e1 = ep[ep.length - 1];
    const X = (e) => L + ((e - e0) / (e1 - e0)) * (W - L - R);
    const Y = (a) => H - B - Math.max(0, Math.min(1, a)) * (H - T - B);

    // 网格与坐标轴
    ctx.lineWidth = 1;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    for (let i = 0; i <= 4; i++) {
      const a = i / 4, y = Y(a);
      ctx.strokeStyle = i === 0 ? "#3a4257" : "#242a38";
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
      ctx.fillStyle = "#7f8ba3";
      ctx.fillText(`${a * 100}%`, 16, y + 4);
    }
    ctx.textAlign = "center";
    const xTicks = [];
    for (let e = 0; e < e1; e += 5000) xTicks.push(e);
    xTicks.push(e1);
    for (const e of xTicks) {
      const x = X(Math.max(e, e0));
      ctx.strokeStyle = "#242a38";
      ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke();
      ctx.fillStyle = "#7f8ba3";
      ctx.fillText(e >= 1000 ? `${Math.round(e / 1000)}k` : `${e}`, x, H - B + 16);
    }

    // 曲线
    const line = (arr, color, width) => {
      if (!arr) return;
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      ep.forEach((e, i) => { const px = X(e), py = Y(arr[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
    };
    line(tr, "#5aa9ff", 1.8);
    line(te, "#ff7a6b", 2.2);

    // 顿悟点
    const grok = grokEpochOf(curve);
    if (grok != null) {
      const gx = X(grok);
      ctx.strokeStyle = "#6ee7a0"; ctx.lineWidth = 1.6;
      ctx.setLineDash([6, 5]);
      ctx.beginPath(); ctx.moveTo(gx, T); ctx.lineTo(gx, H - B); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#6ee7a0"; ctx.textAlign = "left";
      ctx.fillText(`泛化 @ ${grok}`, Math.min(gx + 6, W - R - 76), T + 12);
    }

    // 当前 epoch
    if (state.loadedEpoch != null) {
      const px = X(Math.max(state.loadedEpoch, e0));
      ctx.strokeStyle = "#ffd57a"; ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, H - B); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#ffd57a"; ctx.textAlign = "right";
      ctx.fillText(`${state.loadedEpoch}`, px - 5, H - B - 6);
    }
    ctx.textAlign = "left"; ctx.fillStyle = "#7f8ba3";
    ctx.textAlign = "right"; ctx.fillText("epoch →", W - R, H - 5);

    // 图例文字
    if (legend) {
      const last = (a) => (a && a.length ? (a[a.length - 1] * 100).toFixed(1) + "%" : "—");
      const at = (a, epoch) => {
        if (!a) return "—";
        let best = 0;
        for (let i = 0; i < ep.length && ep[i] <= epoch; i++) best = i;
        return (a[best] * 100).toFixed(1) + "%";
      };
      const st = stageOf(state.loadedEpoch ?? e1);
      legend.innerHTML =
        `<span>实验 <b>${state.run}</b> · seed <b>${state.seed}</b></span>` +
        `<span>P=<b>${state.cfg?.p ?? "—"}</b> · d_model=<b>${state.cfg?.d_model ?? "—"}</b> · heads=<b>${state.cfg?.n_heads ?? "—"}</b> · 训练集占比 <b>${state.cfg?.frac_train ?? "—"}</b></span>` +
        `<span>最终准确率：训练 <b>${last(tr)}</b> · 测试 <b>${last(te)}</b></span>` +
        `<span>顿悟点 <b>${grok ?? "未出现"}</b></span>` +
        `<span>当前 epoch <b>${state.loadedEpoch ?? "—"}</b>（训练 ${at(tr, state.loadedEpoch ?? e1)} / 测试 ${at(te, state.loadedEpoch ?? e1)}，${st || "—"}）</span>`;
    }

    // 交互：点击 / 拖动 → 选最近的快照
    const setFromEvent = (ev) => {
      if (!state.epochs || !state.epochs.length) return;
      const rect = cv.getBoundingClientRect();
      const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
      const px = frac * W;                                   // 换算回画布坐标（去掉两侧留白）
      const epTarget = e0 + ((px - L) / (W - L - R)) * (e1 - e0);
      const nearest = state.epochs.reduce((a, b) =>
        Math.abs(b - epTarget) < Math.abs(a - epTarget) ? b : a, state.epochs[0]);
      $("#epSlider").value = state.epochs.indexOf(nearest);
      selectEpoch(nearest);
    };
    cv.onpointerdown = (ev) => { cv.setPointerCapture(ev.pointerId); setFromEvent(ev); };
    cv.onpointermove = (ev) => { if (ev.buttons & 1) setFromEvent(ev); };
  }

  // -------------------------------------------------------------------
  // 多 seed 横向比较：每个 seed 一小格，点格子即切到该 seed
  // -------------------------------------------------------------------
  // ---------------------------------------------------------------
  // ⓪ 多 seed 横向比较：两种视角
  //    curve = 效果（准确率曲线，数据来自 index.json，零下载）
  //    param = 参数（当前 epoch 的 W_E 频率谱，需要逐个 seed 取权重快照）
  // ---------------------------------------------------------------
  const SEED_NOTE = {
    curve: "每格 = 一个 seed 的准确率曲线（" +
      '<span class="sw" style="background:#4d8fd6"></span>训练集，' +
      '<span class="sw" style="background:#ff7a6b"></span>测试集；绿色虚线 = 顿悟点，金色虚线 = 当前 epoch，只画在选中的格子里）。' +
      "格下方给出该 seed 的顿悟 epoch 与最终测试准确率（青蓝色描边 = 当前选中的 seed）；" +
      "<b>点任意一格即切到该 seed</b>（与工具栏 Seed 下拉、下方曲线卡片联动）。",
    param: "每格 = 该 seed 在<b>当前 epoch</b> 的 W<sub>E</sub> 频率谱（对 token 轴做 DFT，柱高 = Σ<sub>c</sub>|DFT<sub>c</sub>(k)|²，按每格自己归一）。" +
      "5 个 seed 的超参数完全相同，差异只有两处：<b>随机初始化</b>与<b>训练/测试划分</b>——" +
      "结果是每个 seed 都把能量集中到少数几个频率上（稀疏傅里叶电路，集中度 90% 以上），" +
      "但<b>具体是哪些 k 各不相同</b>（实测两两 top-6 只重合 0–2 个，5 个 seed 没有共同频率）。" +
      "格下方标注该 seed 的 top-3 频率与能量占比；换 epoch 会重新取 5 份快照（本地缓存，已取过的不再下载）。" +
      "注：导出时各 seed 保留的快照不一定一样（本 run 里 seed 3/4 只有最终一档），此时会退回到该 seed 最近的快照并标注「最近快照」。",
  };

  function setSeedMode(mode) {
    state.seedMode = mode === "param" ? "param" : "curve";
    for (const b of document.querySelectorAll("#seedModes button")) {
      b.classList.toggle("on", b.dataset.mode === state.seedMode);
    }
    const note = $("#seedNote");
    if (note) note.innerHTML = SEED_NOTE[state.seedMode];
    drawSeeds();
  }

  /** 取某个 (seed, epoch) 的权重快照（带缓存，避免反复下载 887 KB） */
  async function paramSnap(seed, epoch) {
    const key = `${state.run}|${seed}|${epoch}`;
    if (state.snapCache.has(key)) return state.snapCache.get(key);
    const resp = await fetch(`/viz-data/${state.run}/seed${seed}/e${pad6(epoch)}.bin`);
    if (!resp.ok) throw new Error(`seed${seed} e${epoch}: ${resp.status}`);
    const { mats } = parseBin(await resp.arrayBuffer());
    state.snapCache.set(key, mats);
    if (state.snapCache.size > 12) {                      // 简单 FIFO，别把内存撑爆
      state.snapCache.delete(state.snapCache.keys().next().value);
    }
    return mats;
  }

  /** W_E 在前 p 行上的频率能量谱：energy[k] = Σ_c |DFT_c(k)|² */
  function weSpectrum(mats, p) {
    const emb = mats["token_embed"];
    const cols = emb.shape[1];
    const sub = new Float32Array(p * cols);
    for (let r = 0; r < p; r++) {
      for (let c = 0; c < cols; c++) sub[r * cols + c] = emb.data[r * cols + c];
    }
    return fourierNorms1d(sub, p, cols).map((v) => v * v);
  }

  async function drawSeedSpectra(seeds) {
    const gen = ++state.paramGen;
    const p = state.cfg?.p ?? 113;
    const bySeed = state.index?.runs?.[state.run]?.seeds || {};
    const want = state.loadedEpoch ?? (state.epochs?.length ? state.epochs[state.epochs.length - 1] : null);
    const status = $("#seedStatus");
    if (want == null) return;
    let done = 0, fallback = 0;
    for (const s of seeds) {
      const cv = $(`canvas[data-seed="${s}"]`);
      const cap = $(`[data-cap="${s}"]`);
      // 各 seed 导出的快照不全（有的只存了最终一档）→ 退回到该 seed 最近的快照
      const eps = (bySeed[s]?.epochs || []).slice().sort((a, b) => a - b);
      const epoch = eps.length
        ? eps.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a), eps[0])
        : want;
      const exact = epoch === want;
      if (!exact) fallback++;
      if (status) status.textContent = `载入参数快照 ${done}/${seeds.length} …（目标 epoch ${want}）`;
      try {
        const mats = await paramSnap(s, epoch);
        if (gen !== state.paramGen) return;              // 期间又换了 epoch / seed / 模式
        const energy = weSpectrum(mats, p);
        const order = energy.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]);
        const total = energy.reduce((a, b) => a + b, 0) || 1;
        const top3 = order.slice(0, 3).map((e) => e[1]).sort((a, b) => a - b);
        const share = (100 * order.slice(0, 3).reduce((a, e) => a + e[0], 0)) / total;
        drawSeedSpec(cv, energy, top3, s === String(state.seed));
        if (cap) {
          cap.innerHTML = `<div>seed <b>${s}</b> · epoch <b>${epoch}</b>` +
            (exact ? "" : ' <span class="seed-fb">最近快照</span>') + `</div>` +
            `<div>top k: <b>${top3.join(", ")}</b> · 占比 <b>${share.toFixed(1)}%</b></div>`;
        }
      } catch (err) {
        console.error(err);
        if (cap) cap.innerHTML = `<div>seed <b>${s}</b></div><div>快照读取失败</div>`;
      }
      done++;
      if (gen !== state.paramGen) return;
    }
    if (status) {
      status.textContent = `已载入 ${seeds.length} 份快照（epoch ${want}）· 缓存 ${state.snapCache.size} 份` +
        (fallback ? ` · ${fallback} 份用了该 seed 的最近快照` : "");
    }
  }

  /** 一小格：频率能量柱（金色 = W_E 一贯的配色，深色 = 当前选中格里的 top-3） */
  function drawSeedSpec(cv, energy, top3, isCur) {
    if (!cv) return;
    const W = 360, H = 205, L = 26, R = 8, T = 22, B = 18;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0d1017"; ctx.fillRect(0, 0, W, H);
    const n = energy.length;
    const maxE = Math.max(...energy, 1e-12);
    const bw = (W - L - R) / n;
    const topSet = new Set(top3);
    for (let k = 0; k < n; k++) {
      const h = Math.max(0.6, (energy[k] / maxE) * (H - T - B));
      ctx.fillStyle = topSet.has(k) ? (isCur ? "#ffb020" : "#c9a15c") : "#4a3f2a";
      ctx.fillRect(L + k * bw, H - B - h, Math.max(1, bw - 0.8), h);
    }
    ctx.strokeStyle = "#333c50"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, H - B); ctx.lineTo(W - R, H - B); ctx.stroke();
    ctx.fillStyle = "#6b7688"; ctx.font = "10px sans-serif";
    ctx.textAlign = "left"; ctx.fillText("k = 0", L, H - 5);
    ctx.textAlign = "center"; ctx.fillText(`频率 k → ${n - 1}`, (L + W - R) / 2, H - 5);
    ctx.textAlign = "right"; ctx.fillText("归一化能量", W - R, T - 8);
  }

  function drawSeeds() {
    const box = $("#seedRow");
    if (!box) return;
    const bySeed = state.index?.runs?.[state.run]?.seeds;
    if (!bySeed) { box.innerHTML = ""; return; }
    const seeds = Object.keys(bySeed).sort((a, b) => Number(a) - Number(b));
    const mode = state.seedMode || "curve";

    // 格子结构只在 seed 列表变化时重建（重建会重置 canvas）
    if (box.dataset.seeds !== seeds.join(",")) {
      box.dataset.seeds = seeds.join(",");
      box.innerHTML = seeds.map((s) => (
        `<div class="seed-cell" data-seed="${s}">` +
        `<canvas data-seed="${s}" width="360" height="205"></canvas>` +
        `<div class="seed-cap" data-cap="${s}"></div>` +
        `</div>`
      )).join("");
      for (const cell of box.querySelectorAll(".seed-cell")) {
        cell.onclick = () => {
          const s = cell.dataset.seed;
          if (s === String(state.seed)) return;
          $("#seedSel").value = s;
          onSeed();                       // 与工具栏下拉走同一条路径
        };
      }
    }
    const hint = $("#seedsHint");
    if (hint) hint.textContent = `${seeds.length} 个 seed · 点任一格切换`;
    for (const s of seeds) {
      box.querySelector(`.seed-cell[data-seed="${s}"]`)?.classList.toggle("on", s === String(state.seed));
    }

    if (mode === "param") {
      // 先画底色，数据到了再逐格填（见 drawSeedSpectra）
      for (const s of seeds) {
        const cv = $(`canvas[data-seed="${s}"]`);
        if (cv) {
          cv.width = 360; cv.height = 205;
          const ctx = cv.getContext("2d");
          ctx.fillStyle = "#0d1017"; ctx.fillRect(0, 0, 360, 205);
          ctx.fillStyle = "#7f8ba3"; ctx.font = "11px sans-serif";
          ctx.fillText("载入中…", 30, 100);
        }
        const cap = $(`[data-cap="${s}"]`);
        if (cap) cap.innerHTML = `<div>seed <b>${s}</b></div><div>—</div>`;
      }
      drawSeedSpectra(seeds);             // 异步填充
      return;
    }

    for (const s of seeds) {
      const curve = bySeed[s]?.curve;
      drawSeedMini(box.querySelector(`canvas[data-seed="${s}"]`), curve, s === String(state.seed));
      const cap = box.querySelector(`[data-cap="${s}"]`);
      if (cap) {
        const te = curve?.test_acc;
        const fin = te && te.length ? (te[te.length - 1] * 100).toFixed(1) + "%" : "—";
        cap.innerHTML = `<div>seed <b>${s}</b></div>` +
          `<div>泛化 <b>${grokEpochOf(curve) ?? "—"}</b> · 末测 <b>${fin}</b></div>`;
      }
    }
  }

  /** 一小格：单个 seed 的训练/测试准确率曲线（无交互，坐标轴极简） */
  function drawSeedMini(cv, curve, isCur) {
    if (!cv) return;
    const W = 360, H = 205, L = 30, R = 8, T = 10, B = 16;
    cv.width = W; cv.height = H;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0d1017"; ctx.fillRect(0, 0, W, H);

    const ep = curve?.epoch, tr = curve?.train_acc, te = curve?.test_acc;
    if (!ep || !te) {
      ctx.fillStyle = "#7f8ba3"; ctx.font = "11px sans-serif";
      ctx.fillText("无曲线数据", L, H / 2);
      return;
    }
    const e0 = ep[0], e1 = ep[ep.length - 1];
    const X = (e) => L + ((e - e0) / (e1 - e0)) * (W - L - R);
    const Y = (a) => H - B - Math.max(0, Math.min(1, a)) * (H - T - B);

    ctx.font = "10px sans-serif"; ctx.textAlign = "left";
    for (const a of [0, 0.5, 1]) {
      const y = Y(a);
      ctx.strokeStyle = a === 0 ? "#333c50" : "#1e2431";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
      ctx.fillStyle = "#6b7688";
      ctx.fillText(`${a * 100}%`, 3, y + 3.5);
    }
    const line = (arr, color, w) => {
      if (!arr) return;
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.beginPath();
      ep.forEach((e, i) => { const px = X(e), py = Y(arr[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
      ctx.stroke();
    };
    line(tr, "#4d8fd6", 1.3);
    line(te, "#ff7a6b", 1.8);

    const grok = grokEpochOf(curve);
    if (grok != null) {
      ctx.strokeStyle = "#6ee7a0"; ctx.lineWidth = 1.3;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(X(grok), T); ctx.lineTo(X(grok), H - B); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (isCur && state.loadedEpoch != null) {          // 当前 epoch 只画在选中的格子里
      ctx.strokeStyle = "#ffd57a"; ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(X(Math.max(state.loadedEpoch, e0)), T); ctx.lineTo(X(Math.max(state.loadedEpoch, e0)), H - B); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.textAlign = "right"; ctx.fillStyle = "#6b7688";
    ctx.fillText(`${Math.round(e1 / 1000)}k ep`, W - R, H - 4);
  }

  // ===================================================================
  // 视图渲染
  // ===================================================================
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
    drawAcc();
    drawSeeds();
    drawSeeds();
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

  /** 载入某个 epoch 的权重快照。注意：这是**完全自动**的 ——
   *  onSeed / 滑块 / 点击曲线 / 结构图交互都会经由 selectEpoch 调到这里，
   *  工具栏上那个元素只是状态显示（不是按钮）。 */
  async function loadWeights(epoch) {
    const st = $("#loadStatus");
    const setStatus = (text, cls) => {
      if (!st) return;
      st.classList.toggle("busy", cls === "busy");
      st.classList.toggle("err", cls === "err");
      st.innerHTML = (cls === "busy" ? '<span class="spin"></span> ' : "") + text;
    };
    setStatus(`载入 e${epoch}…`, "busy");
    try {
      const file = `e${pad6(epoch)}.bin`;
      const base = `/viz-data/${state.run}/seed${state.seed}/${file}`;
      const resp = await fetch(base);
      if (!resp.ok) throw new Error(`${file} ${resp.status}`);
      const buf = await resp.arrayBuffer();
      const { mats } = parseBin(buf);
      state.mats = mats;
      state.freqPick = {};          // 自动挑中的维度作废，下次打开重新挑
      setStatus(`权重已载入 e${epoch}`);
      buildArch(state.cfg, state.sample);   // 结构图里的“输出张量”要按新权重重画
    } catch (err) {
      console.error(err);
      setStatus(`载入失败 e${epoch}`);
      alert(`载入权重失败：${err.message}`);
    }
  }

  async function onSeed() {
    state.seed = $("#seedSel").value;
    const idx = state.index.runs[state.run].seeds[state.seed];
    state.idx = idx;
    state.cfg = idx.config;
    state.epochs = (idx.epochs || []).slice().sort((a, b) => a - b);
    clearTest();
    drawAddTable();               // 数据划分与 seed 有关（numpy 随机划分），换 seed 要重画
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
    drawAcc();
    drawSeeds();
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
    state.mats = null;            // 换了 seed/run，旧权重作废（结构图会先显示“载入中”）
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

  /** 取一次真实前向的各阶段输出张量（结构图里画的是“经过权重计算以后的结果”）。
   *  没有权重（还没载入）时返回 null，结构图给出占位提示。 */
  function archTensors(a, b) {
    if (!state.mats || !state.cfg) return null;
    const ws = forwardCore(a, b);
    const d = state.cfg.d_model, nh = state.cfg.n_heads;
    const cp = (arr, from) => arr.slice(from, from + d);
    return {
      emb: [cp(ws.x0b, 0), cp(ws.x0b, d), cp(ws.x0b, 2 * d)],        // 3×d：W_E[t] + pos
      heads: Array.from({ length: nh }, (_, i) => cp(ws.headOut, (i * 3 + 2) * d)), // 每个头在「=」处的输出
      attn: cp(ws.acc, 2 * d),                                       // 4 头之和（位置「=」的残差流增量）
      resid2: ws.resid2.slice(),                                     // MLP 的输入
      hid: ws.h.slice(),                                             // ReLU 之后的隐层
      mlpOut: ws.o.slice(),                                          // MLP 输出（+b_out）
      logits: ws.logits.slice(),                                     // 113 类 logit
    };
  }

  /** 一条“张量条带”：以中线为基准，n 个分量画成 n 根细竖条
   *  （正=琥珀向上、负=天蓝向下；高度 ∝ (|v| / vmax)^0.65，幂次压缩是为了让小分量也看得见） */
  function strip(vals, x, y, w, h, vmax) {
    const n = vals.length, bw = w / n, mid = y + h / 2, half = h / 2 - 0.5;
    let t = `<line x1="${x}" y1="${mid.toFixed(1)}" x2="${(x + w).toFixed(1)}" y2="${mid.toFixed(1)}" stroke="#3a4257" stroke-width="0.6"/>`;
    for (let j = 0; j < n; j++) {
      const v = vals[j];
      const bh = Math.max(0.9, Math.pow(Math.min(1, Math.abs(v) / vmax), 0.65) * half);
      const col = v >= 0 ? "#ffb020" : "#5aa9ff";
      const yy = v >= 0 ? mid - bh : mid;
      t += `<rect x="${(x + j * bw).toFixed(2)}" y="${yy.toFixed(2)}" width="${Math.max(0.5, bw - 0.25).toFixed(2)}" height="${bh.toFixed(2)}" fill="${col}"/>`;
    }
    return t;
  }
  /** 若干组数值的绝对值最大（各块各自的归一化基准） */
  const vmaxOf = (lists) => lists.reduce((mx, a) => a.reduce((mm, v) => Math.max(mm, Math.abs(v)), mx), 1e-9);
  /** 单组数值的绝对值最大（在标签里报一下这个条带的实际幅度） */
  const peakOf = (a) => a.reduce((mm, v) => Math.max(mm, Math.abs(v)), 0);

  /** 复用的暂存缓冲：全分辨率网格要跑上万次前向，避免每次新建 ~20 个 TypedArray */
  let _ws = null;
  function workspace(d, dh, nh, nm, p) {
    const key = [d, dh, nh, nm, p].join("|");
    if (_ws && _ws.key === key) return _ws;
    _ws = {
      key,
      x: new Float64Array(3 * d), x0b: new Float64Array(3 * d), acc: new Float64Array(3 * d),
      headOut: new Float64Array(nh * 3 * d),            // 每个头自己的输出（经 W_O）
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
        ws.headOut[(hi * 3 + t) * d + c] = s;              // 同时留一份“这个头自己的输出”
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
    if (!state.mats) { alert("权重还没载入成功（看工具栏右侧的状态），切换一下 Seed 或拖动 epoch 重试即可"); return; }
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
      drawAddTable();                      // 模加法表上给这个样本加白描边
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
  // EMBED 视图：① emb(a) 本体（128 维向量）
  //             ② 这个向量由哪些频率 k 组成
  //             ③ 只用这些频率逆变换回 token 轴 → 复原 one-hot(a)
  //   数学基础（沿 token 轴对每一维做 DFT，恒等变形）：
  //     W_E[a, c] = DC[c] + Σ_k [ cos(2πka/P)·A_k[c] + sin(2πka/P)·B_k[c] ]
  //   A_k / B_k 是长度 d 的“方向向量”（下标 = 维度 c），k = 1…⌊P/2⌋。
  // ===================================================================
  let weBasisCache = { key: null };

  /** 计算并缓存上式的 DC / A_k / B_k（换 run/seed/epoch 才重算） */
  function embedBasis(p, d) {
    const key = `${state.run}|${state.seed}|${state.loadedEpoch}|${p}|${d}`;
    if (weBasisCache.key === key) return weBasisCache;
    const te = state.mats["token_embed"];            // [p+1, d]，行 = token
    const K = Math.floor(p / 2);
    const ct = new Float64Array(p), sn = new Float64Array(p);
    const A = [], B = [];
    for (let k = 1; k <= K; k++) {
      for (let a = 0; a < p; a++) { const th = (2 * Math.PI * k * a) / p; ct[a] = Math.cos(th); sn[a] = Math.sin(th); }
      const Ak = new Float64Array(d), Bk = new Float64Array(d);
      for (let c = 0; c < d; c++) {
        let sa = 0, sb = 0;
        for (let a = 0; a < p; a++) { const v = te.data[a * d + c]; sa += v * ct[a]; sb += v * sn[a]; }
        Ak[c] = (2 * sa) / p; Bk[c] = (2 * sb) / p;
      }
      A.push(Ak); B.push(Bk);
    }
    const DC = new Float64Array(d);
    for (let c = 0; c < d; c++) {
      let s = 0;
      for (let a = 0; a < p; a++) s += te.data[a * d + c];
      DC[c] = s / p;
    }

    // ---- 逆变换用：对基矩阵做 QR 分解（列 = DC, A_1, B_1, A_2, B_2, …，共 nb = 2K+1 = 113 列）----
    // 有了 Q、R，就能从 emb(a) 的 d 个数解出 nb 个频率系数 x = [1, cosθ_1, sinθ_1, …]：
    //     x = R⁻¹ · (Qᵀ · w)        （这就是「逆变换」的第一步）
    // 用修正 Gram-Schmidt（比正规方程 Bm·Bmᵀ 稳定得多：后者会把条件数平方到 1.4e10）
    const nb = 2 * K + 1;
    const bcols = [DC];
    for (let k = 0; k < K; k++) { bcols.push(A[k]); bcols.push(B[k]); }
    const Qc = [], Rm = new Float64Array(nb * nb);
    for (let j = 0; j < nb; j++) {
      const v = Float64Array.from(bcols[j]);
      for (let i = 0; i < j; i++) {
        const q = Qc[i];
        let s = 0;
        for (let t = 0; t < d; t++) s += q[t] * v[t];
        Rm[i * nb + j] = s;
        for (let t = 0; t < d; t++) v[t] -= s * q[t];
      }
      let nrm = 0;
      for (let t = 0; t < d; t++) nrm += v[t] * v[t];
      nrm = Math.sqrt(nrm);
      Rm[j * nb + j] = nrm;
      const q = new Float64Array(d);
      if (nrm > 1e-300) for (let t = 0; t < d; t++) q[t] = v[t] / nrm;
      Qc.push(q);
    }

    weBasisCache = { key, A, B, DC, K, p, d, Qc, Rm, nb };
    return weBasisCache;
  }

  /** 真实逆变换第 1 步：把 emb(a) 的 d 个数解成 nb 个频率系数（x[0] 应 ≈ 1，x[2k−1]/x[2k] 应 ≈ cos/sin θ_k） */
  function embedCoeffs(bs, w) {
    const { Qc, Rm, nb } = bs;
    const u = new Float64Array(nb);
    for (let j = 0; j < nb; j++) {
      const q = Qc[j];
      let s = 0;
      for (let t = 0; t < w.length; t++) s += q[t] * w[t];
      u[j] = s;
    }
    const x = new Float64Array(nb);
    for (let j = nb - 1; j >= 0; j--) {
      let s = u[j];
      for (let i = j + 1; i < nb; i++) s -= Rm[j * nb + i] * x[i];
      const dg = Rm[j * nb + j];
      x[j] = Math.abs(dg) > 1e-300 ? s / dg : 0;
    }
    return x;
  }

  /** ① emb(a) 本体：横轴 = 维度下标 1…d，纵轴 = 数值（琥珀 = 正，天蓝 = 负）
   *  底部色带 = 每个维度的“主导频率”（该维振幅最大的 k），用来标出关键频率落在哪些维度上 */
  function drawEmbedVector(canvas, d, a, vec, bs) {
    const W = 1100, H = 362;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const L = 78, R = 18, T = 30, B = 100;
    const pw = W - L - R, ph = H - T - B;
    let vmax = 0, vmin = 0, iMax = 0;
    for (let c = 0; c < d; c++) {
      if (vec[c] > vmax) { vmax = vec[c]; iMax = c; }
      if (vec[c] < vmin) vmin = vec[c];
    }
    const m = Math.max(vmax, -vmin, 1e-12);
    const mid = T + ph / 2;
    const Y = (v) => mid - ((ph / 2) * 0.92 * v) / m;
    const bw = pw / d;
    const X = (c) => L + c * bw;
    const fmt = (v) => (v === 0 ? "0" : Math.abs(v) < 0.01 ? v.toExponential(1) : v.toFixed(3));

    // ---- 每个维度的主导频率：amp_k[c] = √(A_k[c]² + B_k[c]²)，取最大的 k ----
    const K = bs.K;
    const ampK = [];
    for (let k = 0; k < K; k++) {
      const Ak = bs.A[k], Bk = bs.B[k], arr = new Float64Array(d);
      for (let c = 0; c < d; c++) arr[c] = Math.hypot(Ak[c], Bk[c]);
      ampK.push(arr);
    }
    const totE = new Float64Array(d);
    for (let k = 0; k < K; k++) { const arr = ampK[k]; for (let c = 0; c < d; c++) totE[c] += arr[c] * arr[c]; }
    const dom = new Int32Array(d), domShare = new Float64Array(d);
    const cnt = new Map();
    for (let c = 0; c < d; c++) {
      let bk = 1, bv = -1;
      for (let k = 0; k < K; k++) { const v = ampK[k][c]; if (v > bv) { bv = v; bk = k + 1; } }
      dom[c] = bk; domShare[c] = totE[c] > 0 ? (bv * bv) / totE[c] : 0;
      cnt.set(bk, (cnt.get(bk) || 0) + 1);
    }
    const ranked = [...cnt.entries()].sort((x, z) => z[1] - x[1] || x[0] - z[0]);
    const PAL = ["#c1440e", "#2e8b66", "#7b5bd6"], OTHER = "#cfd6e2";
    const domColor = new Map();
    ranked.slice(0, 3).forEach(([k], i) => domColor.set(k, PAL[i]));

    // 网格 + 0 基线
    ctx.strokeStyle = "#eef0f4"; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const gy = T + (ph * g) / 4;
      ctx.beginPath(); ctx.moveTo(L, gy); ctx.lineTo(W - R, gy); ctx.stroke();
    }
    ctx.strokeStyle = "#c9ced7"; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(W - R, Y(0)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    ctx.fillText(`+${fmt(m)}`, L - 8, Y(m) + 4);
    ctx.fillText("0", L - 8, Y(0) + 4);
    ctx.fillText(`−${fmt(m)}`, L - 8, Y(-m) + 4);

    // 柱子：emb(a) 的每个分量
    for (let c = 0; c < d; c++) {
      const v = vec[c];
      const y0 = Y(0), y1 = Y(v);
      ctx.fillStyle = v >= 0 ? "#d98a1f" : "#4d8fd6";
      ctx.fillRect(X(c) + 0.6, Math.min(y0, y1), Math.max(bw - 1.2, 0.8), Math.max(Math.abs(y1 - y0), 1));
    }

    // ---- 色带：每一格 = 该维度的主导频率 ----
    const ry = T + ph + 10, rh = 11;
    for (let c = 0; c < d; c++) {
      ctx.fillStyle = domColor.get(dom[c]) || OTHER;
      ctx.fillRect(X(c) + 0.4, ry, Math.max(bw - 0.8, 0.9), rh);
    }
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif"; ctx.textAlign = "right";
    ctx.fillText("主导频率", L - 6, ry + rh - 1.5);

    // ---- 标记：前 3 个主导频率各自振幅最大的那个维度 ----
    const marks = ranked.slice(0, 3)
      .filter(([, n]) => n >= 3)
      .map(([k], i) => {
        let bc = 0, bv = -1;
        for (let c = 0; c < d; c++) { const v = ampK[k - 1][c]; if (v > bv) { bv = v; bc = c; } }
        return { k, c: bc, col: PAL[i], amp: bv };
      });
    for (const mk of marks) {
      const xc = X(mk.c) + bw / 2;
      ctx.fillStyle = mk.col;
      ctx.beginPath();
      ctx.moveTo(xc, ry - 2); ctx.lineTo(xc - 5, ry - 10); ctx.lineTo(xc + 5, ry - 10);
      ctx.closePath(); ctx.fill();
      const label = `k=${mk.k}`;
      ctx.font = "bold 10.5px monospace";
      const lw = ctx.measureText(label).width;
      let lx = Math.max(L + lw / 2 + 2, Math.min(W - R - lw / 2 - 2, xc));
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(lx - lw / 2 - 3, ry - 27, lw + 6, 14);
      ctx.fillStyle = mk.col; ctx.textAlign = "center";
      ctx.fillText(label, lx, ry - 16);
    }

    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    ctx.fillText(`emb(a = ${a}) 本体：W_E[${a}, :]，全部 ${d} 维 · 琥珀 = 正 / 天蓝 = 负`, L, 17);
    ctx.textAlign = "right"; ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
    ctx.fillText(`最大 |值| = ${fmt(m)}（dim ${iMax + 1}）· 柱子看不出周期，用色带看频率归属`, W - R, 17);

    ctx.textAlign = "center"; ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
    for (let c = 0; c < d; c += 16) {
      const isLast = c + 16 >= d;
      ctx.textAlign = isLast ? "right" : "center";
      ctx.fillText(String(c + 1), isLast ? W - R : X(c) + bw / 2, T + ph + 46);
    }

    // ---- 图例 ----
    ctx.textAlign = "left"; ctx.fillStyle = "#333"; ctx.font = "12px sans-serif";
    ctx.fillText("色带 = 该维的主导频率（振幅最大的 k）：", L, H - 22);
    let lx2 = L + 244;
    const chip = (col, label) => {
      ctx.fillStyle = col; ctx.fillRect(lx2, H - 32, 10, 10);
      ctx.fillStyle = "#57606a"; ctx.font = "11.5px sans-serif"; ctx.textAlign = "left";
      ctx.fillText(label, lx2 + 14, H - 23);
      lx2 += 14 + ctx.measureText(label).width + 16;
    };
    let covered = 0;
    for (const [k, n] of ranked.slice(0, 3)) { chip(domColor.get(k), `k=${k}（${n} 维）`); covered += n; }
    chip(OTHER, `其它（${d - covered} 维）`);
    ctx.textAlign = "right"; ctx.fillStyle = "#333"; ctx.font = "12px sans-serif";
    ctx.fillText("维度下标 c（1…128）→", W - R, H - 8);

    // 供文案使用的统计量：主导频率占该维能量的中位数、前 6 个频率合计占比中位数
    const med = (arr) => { const s = Array.from(arr).sort((x, z) => x - z); return s[s.length >> 1] || 0; };
    const t6 = new Float64Array(d);
    for (const [k] of ranked.slice(0, 6)) {
      const arr = ampK[k - 1];
      for (let c = 0; c < d; c++) t6[c] += arr[c] * arr[c];
    }
    return {
      ranked, covered, marks,
      medShare: med(domShare),
      medTop6: med(Array.from(t6, (v, c) => (totE[c] > 0 ? v / totE[c] : 0))),
    };
  }

  /** ② 56 个频率各自长什么样：每条波 = cos(2πk(n−a)/P)，2 列 × 28 行
   *  右侧数字 = 该频率在 emb(a) 里的真实强度 R（由权重算得） */
  function drawEmbedWaves(canvas, p, a, K, amps) {
    const W = 1100, nCols = 2, gx = 30, cellH = 27, topPad = 58, botPad = 26;
    const nRows = Math.ceil(K / nCols);
    const H = topPad + nRows * cellH + botPad;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const cw = (W - gx * (nCols - 1)) / nCols;
    let amax = 1e-12;
    for (let i = 0; i < K; i++) amax = Math.max(amax, amps[i]);
    const padL = 46, padR = 66;

    for (let i = 0; i < K; i++) {
      const k = i + 1;
      const ox = ((i / nRows) | 0) * (cw + gx);
      const cy = topPad + ((i % nRows) | 0) * cellH + cellH / 2;
      const px = ox + padL, pw = cw - padL - padR;
      const X = (n) => px + (pw * n) / (p - 1);
      const A = 9;
      ctx.strokeStyle = "#eceff4"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, cy); ctx.lineTo(px + pw, cy); ctx.stroke();
      const hue = 205 + (i / Math.max(1, K - 1)) * 135;
      ctx.strokeStyle = `hsl(${hue}, 72%, 50%)`; ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let n = 0; n < p; n++) {
        const v = Math.cos((2 * Math.PI * k * (n - a)) / p);
        const x = X(n), y = cy - v * A;
        n ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
      ctx.fillStyle = "#57606a"; ctx.font = "10.5px monospace"; ctx.textAlign = "right";
      ctx.fillText(`k=${k}`, px - 6, cy + 3.5);
      const strong = amps[i] >= 0.2 * amax;
      ctx.fillStyle = strong ? "#b0332f" : "#9aa3b2";
      ctx.font = strong ? "bold 10.5px monospace" : "10.5px monospace";
      ctx.textAlign = "left";
      ctx.fillText(amps[i].toFixed(3), px + pw + 6, cy + 3.5);
    }
    // 每列一条 n = a 的绿虚线：所有波都在这里同相
    for (let col = 0; col < nCols; col++) {
      const ox = col * (cw + gx);
      const px = ox + padL, pw = cw - padL - padR;
      const xa = px + (pw * a) / (p - 1);
      ctx.strokeStyle = "rgba(46,139,102,.75)"; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(xa, topPad - 8); ctx.lineTo(xa, topPad + nRows * cellH); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px monospace"; ctx.textAlign = "center";
      ctx.fillText("n = 0", px, topPad + nRows * cellH + 14);
      ctx.fillText(`n = a = ${a}`, xa, topPad - 14);
      ctx.fillText(`n = ${p - 1}`, px + pw, topPad + nRows * cellH + 14);
    }
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    ctx.fillText(`${K} 个频率各自的样子：每条波 = cos(2πk(n−a)/P)（113 步转 k 圈）`, 20, 18);
    ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
    ctx.fillText("右列 R = 该频率那一份的长度 ‖v_k‖（128 维模长，不是某一维的取值）· 绿虚线 = n = a：所有波在这里同相", 20, 33);
  }

  /** ③ 把频率叠加回 token 轴：J 个频率的和（J = 4 / 16 / 全部 56）→ 归一化后就是 one-hot(a) */
  function drawEmbedSum(canvas, p, a, K) {
    const W = 1100, H = 360;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const L = 76, R = 22, T = 58, B = 46;
    const pw = W - L - R, ph = H - T - B;
    const X = (n) => L + (pw * n) / (p - 1);
    const vmin = -0.03, vmax = 1.06;
    const Y = (v) => T + ph - (ph * (Math.max(vmin, Math.min(vmax, v)) - vmin)) / (vmax - vmin);
    const sum = (J) => {
      const out = new Float64Array(p);
      for (let n = 0; n < p; n++) {
        let s = 0;
        for (let k = 1; k <= J; k++) s += Math.cos((2 * Math.PI * k * (n - a)) / p);
        out[n] = (s + 0.5) / (J + 0.5);         // 归一化：峰高恒为 1；J = K 时正好是 one-hot
      }
      return out;
    };

    // 网格 + 0/1 刻度
    ctx.strokeStyle = "#eef0f4"; ctx.lineWidth = 1;
    for (const g of [0, 0.25, 0.5, 0.75, 1]) {
      const gy = Y(g);
      ctx.beginPath(); ctx.moveTo(L, gy); ctx.lineTo(W - R, gy); ctx.stroke();
    }
    ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    for (const g of [0, 0.5, 1]) ctx.fillText(g.toFixed(1), L - 8, Y(g) + 4);

    // one-hot 参考（灰柱）
    const bw = pw / p;
    for (let n = 0; n < p; n++) {
      if (n !== a) continue;
      ctx.fillStyle = "#c8d1e0";
      ctx.fillRect(X(n) - bw / 2, Y(1), Math.max(bw, 3), Y(0) - Y(1));
    }
    // n = a 竖线
    ctx.strokeStyle = "rgba(46,139,102,.8)"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(X(a), T - 26); ctx.lineTo(X(a), T + ph + 6); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#2e8b66"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(`n = a = ${a}`, X(a), T - 12);

    // 三条曲线
    const line = (arr, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      for (let n = 0; n < p; n++) { const x = X(n), y = Y(arr[n]); n ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    };
    line(sum(4), "#f0b872", 1.6);
    line(sum(16), "#e08b3a", 1.9);
    line(sum(K), "#c1440e", 2.4);

    // 标题 + 图例
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    ctx.fillText("把频率一条条叠回去（逆变换）", L, 18);
    let lx = L + ctx.measureText("把频率一条条叠回去（逆变换）").width + 18;
    const sw = (color, label, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(lx, 13); ctx.lineTo(lx + 20, 13); ctx.stroke();
      ctx.fillStyle = "#57606a"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
      ctx.fillText(label, lx + 25, 17);
      lx += 25 + ctx.measureText(label).width + 14;
    };
    sw("#f0b872", "前 4 个频率", 1.6);
    sw("#e08b3a", "前 16 个", 1.9);
    sw("#c1440e", `全部 ${K} 个（= one-hot）`, 2.4);
    ctx.fillStyle = "#c8d1e0"; ctx.fillRect(lx, 8, 8, 10);
    ctx.fillStyle = "#57606a"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
    ctx.fillText("one-hot(a) 参考", lx + 13, 17);

    // x 轴
    ctx.textAlign = "center"; ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
    for (let n = 0; n < p; n += 16) {
      const isLast = n + 16 >= p;
      ctx.textAlign = isLast ? "right" : "center";
      ctx.fillText(String(n), isLast ? W - R : X(n), T + ph + 16);
    }
    ctx.textAlign = "right"; ctx.fillStyle = "#333"; ctx.font = "12px sans-serif";
    ctx.fillText("token n →", W - R, H - 8);
  }

  /** ④ 真实逆变换：真的用 emb(a) 的 128 个数算出整条曲线
   *  红线 = 全 113 个系数（精确等于 one-hot(a)）；橙线 = 只保留最强 J 个频率（相位对，但旁瓣大） */
  function drawEmbedInverse(canvas, p, a, K, yFull, yJ, J) {
    const W = 1100, H = 366;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const L = 76, R = 22, T = 60, B = 48;
    const pw = W - L - R, ph = H - T - B;
    const X = (n) => L + (pw * n) / (p - 1);
    const vmin = -0.22, vmax = 1.14;
    const Y = (v) => T + ph - (ph * (Math.max(vmin, Math.min(vmax, v)) - vmin)) / (vmax - vmin);

    // 网格 + 刻度
    ctx.strokeStyle = "#eef0f4"; ctx.lineWidth = 1;
    for (const g of [0, 0.25, 0.5, 0.75, 1]) {
      ctx.beginPath(); ctx.moveTo(L, Y(g)); ctx.lineTo(W - R, Y(g)); ctx.stroke();
    }
    ctx.strokeStyle = "#dfe3ea"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(W - R, Y(0)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    for (const g of [0, 0.5, 1]) ctx.fillText(g.toFixed(1), L - 8, Y(g) + 4);

    // one-hot(a) 参考柱
    const bw = pw / p;
    ctx.fillStyle = "#c8d1e0";
    ctx.fillRect(X(a) - Math.max(bw, 3) / 2, Y(1), Math.max(bw, 3), Y(0) - Y(1));
    // n = a 竖线
    ctx.strokeStyle = "rgba(46,139,102,.8)"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(X(a), T - 26); ctx.lineTo(X(a), T + ph + 6); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#2e8b66"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "center";
    ctx.fillText(`n = a = ${a}`, X(a), T - 12);

    const line = (arr, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      for (let n = 0; n < p; n++) { const x = X(n), y = Y(arr[n]); n ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    };
    line(yJ, "#e08b3a", 1.9);
    line(yFull, "#c1440e", 2.4);

    // 标题 + 图例
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    const title = "④ 真实逆变换：直接拿 emb(a) 的 128 个数算出这条曲线";
    ctx.fillText(title, L, 18);
    ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
    ctx.fillText("横轴 = 113 个 token n；纵轴 = 复原出来的「是 token n 吗」（灰柱 = 真值 one-hot(a)）", L, 34);
    let lx = L + ctx.measureText("横轴 = 113 个 token n；纵轴 = 复原出来的「是 token n 吗」（灰柱 = 真值 one-hot(a)）").width + 16;
    const sw = (color, label, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(lx, 30); ctx.lineTo(lx + 20, 30); ctx.stroke();
      ctx.fillStyle = "#57606a"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
      ctx.fillText(label, lx + 25, 34);
      lx += 25 + ctx.measureText(label).width + 14;
    };
    sw("#c1440e", `全部 ${K} 个频率（精确 = one-hot）`, 2.4);
    sw("#e08b3a", `只用最强 ${J} 个频率`, 1.9);

    // x 轴
    ctx.textAlign = "center"; ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
    for (let n = 0; n < p; n += 16) {
      const isLast = n + 16 >= p;
      ctx.textAlign = isLast ? "right" : "center";
      ctx.fillText(String(n), isLast ? W - R : X(n), T + ph + 16);
    }
    ctx.textAlign = "right"; ctx.fillStyle = "#333"; ctx.font = "12px sans-serif";
    ctx.fillText("token n →", W - R, H - 8);
  }

  /** ⓿ 纯 DFT：one-hot(a) 的完整往返（与权重无关，只用到 a）
   *  左上 (a) 时域 ──DFT──▶ 左下 (c) 幅度谱 / 右下 (d) 相位谱 ──逆DFT──▶ 右上 (b) 又回到时域 */
  function drawEmbedDft(canvas, p, a, K) {
    const W = 1100, H = 322;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const L = 78, R = 22, pw = W - L - R, gapX = 44, colW = (pw - gapX) / 2;
    const xA = L, xB = L + colW + gapX;
    const Rk = 2 / p, c0 = 1 / p;
    const vmin = -0.16, vmax = 1.08;
    const YT = (y0, h, v) => y0 + h - (h * (Math.max(vmin, Math.min(vmax, v)) - vmin)) / (vmax - vmin);

    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12.5px sans-serif";
    ctx.fillText("⓿ 纯 DFT：把 one-hot(a) 从时域变到频域，再用逆 DFT 变回来（全程与权重无关，只用到 a）", L, 18);
    ctx.fillStyle = "#8a93a5"; ctx.font = "11px sans-serif";
    ctx.fillText(`(a) 时域 ──DFT──▶ (c) 幅度谱 + (d) 相位谱 ──逆DFT──▶ (b) 又回到时域；` +
      `进去 ${p} 个数，出来还是 ${p} 个数（1 个直流 + ${K} 对 cos/sin）`, L, 34);

    // ============ (a) 左上：时域 ============
    const yA = 58, hA = 86;
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    ctx.fillText(`(a) 时域：one-hot(a)`, xA, 52);
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif";
    ctx.fillText(`n = 0…${p - 1}`, xA + 118, 52);
    ctx.strokeStyle = "#eef0f4"; ctx.lineWidth = 1;
    for (const g of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(xA, YT(yA, hA, g)); ctx.lineTo(xA + colW, YT(yA, hA, g)); ctx.stroke(); }
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif"; ctx.textAlign = "right";
    ctx.fillText("1", xA - 6, YT(yA, hA, 1) + 4); ctx.fillText("0", xA - 6, YT(yA, hA, 0) + 4);
    const bw1 = colW / p;
    ctx.fillStyle = "#cfd6e2";
    ctx.fillRect(xA, YT(yA, hA, 0.012), colW, yA + hA - YT(yA, hA, 0.012));
    ctx.fillStyle = "#c1440e";
    ctx.fillRect(xA + a * bw1 - Math.max(bw1, 3) / 2, YT(yA, hA, 1), Math.max(bw1, 3), yA + hA - YT(yA, hA, 1));
    ctx.fillStyle = "#c1440e"; ctx.font = "bold 10.5px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(`n = a = ${a} → 1`, Math.min(xA + a * bw1 + 6, xA + colW - 74), yA + 14);

    // ============ (b) 右上：逆 DFT 回来 ============
    const yB = 58, hB = 86;
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    ctx.fillText(`(b) 逆 DFT 回来：又是 one-hot(a)`, xB, 52);
    ctx.strokeStyle = "#eef0f4"; ctx.lineWidth = 1;
    for (const g of [0, 0.5, 1]) { ctx.beginPath(); ctx.moveTo(xB, YT(yB, hB, g)); ctx.lineTo(xB + colW, YT(yB, hB, g)); ctx.stroke(); }
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif"; ctx.textAlign = "right";
    ctx.fillText("1", xB - 6, YT(yB, hB, 1) + 4); ctx.fillText("0", xB - 6, YT(yB, hB, 0) + 4);
    ctx.fillStyle = "#cfd6e2";
    ctx.fillRect(xB + a * bw1 - Math.max(bw1, 3) / 2, YT(yB, hB, 1), Math.max(bw1, 3), yB + hB - YT(yB, hB, 1));

    // 频域系数（实 DFT）：a_k = (2/P)cos(2πka/P)，b_k = (2/P)sin(2πka/P)
    const Ack = new Float64Array(K + 1), Bck = new Float64Array(K + 1);
    for (let k = 1; k <= K; k++) {
      const th = (2 * Math.PI * k * a) / p;
      Ack[k] = (2 / p) * Math.cos(th);
      Bck[k] = (2 / p) * Math.sin(th);
    }
    const invDft = (J) => {
      const out = new Float64Array(p);
      for (let n = 0; n < p; n++) {
        let s = c0;
        for (let k = 1; k <= J; k++) s += Ack[k] * Math.cos((2 * Math.PI * k * n) / p) + Bck[k] * Math.sin((2 * Math.PI * k * n) / p);
        out[n] = s;
      }
      return out;
    };
    const yFull = invDft(K), y16 = invDft(16);
    let rtErr = 0;
    for (let n = 0; n < p; n++) rtErr = Math.max(rtErr, Math.abs(yFull[n] - (n === a ? 1 : 0)));
    const XT = (x0, n) => x0 + (colW * n) / (p - 1);
    ctx.strokeStyle = "#f0c08a"; ctx.lineWidth = 1.4; ctx.beginPath();
    for (let n = 0; n < p; n++) { const x = XT(xB, n), y = YT(yB, hB, y16[n]); n ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
    ctx.strokeStyle = "#c1440e"; ctx.lineWidth = 1.8; ctx.beginPath();
    for (let n = 0; n < p; n++) { const x = XT(xB, n), y = YT(yB, hB, yFull[n]); n ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
    ctx.fillStyle = "#b3761a"; ctx.font = "10.5px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(`前 16 个频率（峰 ${y16[a].toFixed(2)}）`, xB + 4, yB + 14);
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(`与 (a) 逐点重合，最大偏差 ${rtErr.toExponential(0)}`, xB + 206, 52);
    ctx.fillStyle = "#c1440e"; ctx.font = "bold 10.5px sans-serif"; ctx.textAlign = "right";
    ctx.fillText(`全部 ${K} 个 → 峰 1.000`, xB + colW - 4, yB + 14);

    // 时域两图的 x 刻度
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif"; ctx.textAlign = "center";
    for (const x0 of [xA, xB]) {
      for (let n = 0; n < p; n += 16) {
        const isLast = n + 16 >= p;
        ctx.textAlign = isLast ? "right" : "center";
        ctx.fillText(String(n), isLast ? x0 + colW : XT(x0, n), yA + hA + 15);
      }
    }

    // ============ (c) 左下：幅度谱 ============
    const yC = 196, hC = 92;
    const YC = (v) => yC + hC - (hC * v) / (Rk * 1.3);
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    ctx.fillText(`(c) 幅度谱：|c_k|（k = 1…${K}）`, xA, 186);
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif";
    ctx.fillText(`全平，且与 a 无关`, xA + 176, 186);
    ctx.strokeStyle = "#eef0f4";
    for (const g of [0, c0, Rk]) { ctx.beginPath(); ctx.moveTo(xA, YC(g)); ctx.lineTo(xA + colW, YC(g)); ctx.stroke(); }
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif"; ctx.textAlign = "right";
    ctx.fillText("2/113", xA - 6, YC(Rk) + 4); ctx.fillText("0", xA - 6, YC(0) + 4);
    const bwK = colW / K;
    for (let i = 0; i < K; i++) {
      ctx.fillStyle = "#2e8b66";
      ctx.fillRect(xA + i * bwK + 0.8, YC(Rk), Math.max(bwK - 1.6, 1.0), YC(0) - YC(Rk));
    }
    ctx.strokeStyle = "#b8860b"; ctx.setLineDash([4, 3]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(xA, YC(c0)); ctx.lineTo(xA + colW, YC(c0)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#b3761a"; ctx.font = "bold 10.5px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(`每个 k 都是 2/${p} = ${Rk.toFixed(5)}（虚线 = 直流 ${c0.toFixed(5)}，正好一半）`, xA + 4, yC + 14);

    // ============ (d) 右下：相位谱 ============
    const yD = 196, hD = 92;
    const stepRad = (2 * Math.PI * a) / p;
    const YD = (v) => yD + hD - (hD * v) / (2 * Math.PI);
    ctx.textAlign = "left"; ctx.fillStyle = "#2a2f3a"; ctx.font = "bold 12px sans-serif";
    ctx.fillText(`(d) 相位谱：∠c_k = −2πka/${p}（对 2π 取模）`, xB, 186);
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif";
    ctx.fillText(`每步前进 2πa/${p} = ${stepRad.toFixed(3)} rad`, xB + 268, 186);
    ctx.strokeStyle = "#eef0f4";
    for (const g of [0, Math.PI, 2 * Math.PI]) { ctx.beginPath(); ctx.moveTo(xB, YD(g)); ctx.lineTo(xB + colW, YD(g)); ctx.stroke(); }
    ctx.fillStyle = "#8a93a5"; ctx.textAlign = "right";
    ctx.fillText("2π", xB - 6, YD(2 * Math.PI) + 4);
    ctx.fillText("π", xB - 6, YD(Math.PI) + 4);
    ctx.fillText("0", xB - 6, YD(0) + 4);
    const XD = (k) => xB + ((k - 0.5) * colW) / K;
    const phOf = (k) => ((2 * Math.PI * k * a) / p) % (2 * Math.PI);
    ctx.strokeStyle = "#e6d2b0"; ctx.lineWidth = 1.2; ctx.beginPath();
    for (let k = 1; k <= K; k++) {
      const y = YD(phOf(k));
      k === 1 ? ctx.moveTo(XD(k), y) : ctx.lineTo(XD(k), y);
    }
    ctx.stroke();
    for (let k = 1; k <= K; k++) {
      ctx.fillStyle = "#b8860b";
      ctx.beginPath(); ctx.arc(XD(k), YD(phOf(k)), 1.4, 0, 2 * Math.PI); ctx.fill();
    }

    // 频域两图的 x 刻度
    ctx.fillStyle = "#8a93a5"; ctx.font = "10.5px sans-serif";
    for (const [x0] of [[xA], [xB]]) {
      for (const k of [1, 16, 32, 48, K]) {
        ctx.textAlign = k === K ? "right" : "center";
        ctx.fillText(String(k), k === K ? x0 + colW : x0 + ((k - 0.5) * colW) / K, yC + hC + 15);
      }
    }
  }

  /** EMBED 视图主流程：⓿ one-hot(a) 的 DFT ① emb(a) 本体 ② 56 个频率各自的样子 ③ 叠加回去 = one-hot(a) ④ 真实逆变换 */
  function runEmbedView(a) {
    const p = state.cfg.p, d = state.cfg.d_model;
    const te = state.mats["token_embed"];
    const bs = embedBasis(p, d);
    const K = bs.K;                                   // = ⌊P/2⌋ = 56

    // —— ① emb(a) 本体 ——
    const vec = new Float64Array(d);
    for (let c = 0; c < d; c++) vec[c] = te.data[a * d + c];
    const dv = drawEmbedVector($("#cEmbedWave"), d, a, vec, bs);

    // —— ② 每个频率的强度（由真实权重算）+ 56 条波长什么样 ——
    const amps = new Float64Array(K);
    for (let k = 1; k <= K; k++) {
      const th = (2 * Math.PI * k * a) / p, ck = Math.cos(th), sk = Math.sin(th);
      const Ak = bs.A[k - 1], Bk = bs.B[k - 1];
      let s = 0;
      for (let c = 0; c < d; c++) { const t = ck * Ak[c] + sk * Bk[c]; s += t * t; }
      amps[k - 1] = Math.sqrt(s);
    }
    drawEmbedWaves($("#cEmbedSplit"), p, a, K, amps);

    // —— ⓿ 纯 DFT：one-hot(a) 从时域变到频域（与权重无关，只用到 a）——
    drawEmbedDft($("#cEmbedDft"), p, a, K);

    // —— ③ 叠加回 token 轴 ——
    drawEmbedSum($("#cEmbedParts"), p, a, K);

    // 自检：用这 56 个频率与它们的相位重建 emb(a)，误差应为机器精度
    let sse = 0, sst = 0;
    for (let c = 0; c < d; c++) {
      let s = bs.DC[c];
      for (let k = 1; k <= K; k++) {
        const th = (2 * Math.PI * k * a) / p;
        s += Math.cos(th) * bs.A[k - 1][c] + Math.sin(th) * bs.B[k - 1][c];
      }
      sse += (vec[c] - s) * (vec[c] - s);
      sst += vec[c] * vec[c];
    }
    const errPct = sst > 0 ? (100 * sse) / sst : 0;
    const top = Array.from({ length: K }, (_, i) => i + 1).sort((x, z) => amps[z - 1] - amps[x - 1]);
    const a1 = top[0], a2 = top[1], a3 = top[2];

    // —— ④ 真实逆变换：emb(a) 的 128 个数 → 113 个系数 x → 整条 one-hot 曲线 ——
    const xc = embedCoeffs(bs, vec);
    const J6 = Math.min(6, K);
    const keep = new Uint8Array(K + 1);
    for (let i = 0; i < J6; i++) keep[top[i]] = 1;
    const yFull = new Float64Array(p), yJ = new Float64Array(p);
    for (let n = 0; n < p; n++) {
      let sf = xc[0], sj = xc[0];
      for (let k = 1; k <= K; k++) {
        const th = (2 * Math.PI * k * n) / p;
        const v = xc[2 * k - 1] * Math.cos(th) + xc[2 * k] * Math.sin(th);
        sf += v;
        if (keep[k]) sj += v;
      }
      yFull[n] = (sf - 0.5) / (K + 0.5);
      yJ[n] = (sj - 0.5) / (J6 + 0.5);
    }
    drawEmbedInverse($("#cEmbedInv"), p, a, K, yFull, yJ, J6);

    // 逆变换的自检量
    let eCoeff = Math.abs(xc[0] - 1);
    for (let k = 1; k <= K; k++) {
      const th = (2 * Math.PI * k * a) / p;
      eCoeff = Math.max(eCoeff, Math.abs(xc[2 * k - 1] - Math.cos(th)), Math.abs(xc[2 * k] - Math.sin(th)));
    }
    let eHot = 0, okArg = true, peak = 0;
    for (let n = 0; n < p; n++) {
      const truth = n === a ? 1 : 0;
      eHot = Math.max(eHot, Math.abs(yFull[n] - truth));
      if (yFull[n] > peak) { peak = yFull[n]; okArg = n === a; }
    }
    let jPeak = 0, jArg = -1, jSecond = 0;
    for (let n = 0; n < p; n++) if (yJ[n] > jPeak) { jPeak = yJ[n]; jArg = n; }
    for (let n = 0; n < p; n++) if (n !== jArg && yJ[n] > jSecond) jSecond = yJ[n];

    // EMBED 只有三块：本体 / 56 个频率 / 叠加；其它视图用的那块先藏起来
    for (const id of ["#freqCap2", "#cEmbedSpec", "#edSpecNote"]) $(id).hidden = true;
    const nSel = $("#freqTop");
    if (nSel && nSel.closest(".kv")) nSel.closest(".kv").style.display = "none";
    $("#embedSplitWrap").hidden = false;
    $("#embedDftWrap").hidden = false;

    // —— 文案 ——
    const st = stageOf(state.loadedEpoch);
    $("#freqStage").textContent = `${state.loadedEpoch} 步${st ? " · " + st : ""}`;
    $("#freqCap0").innerHTML =
      `⓿ <b>纯 DFT 往返</b>：one-hot(a) 时域 →(DFT)→ 频域 →(逆 DFT)→ 又回到时域 —— ` +
      `左上 = 输入，左下 / 右下 = 中间步骤，右上 = 输出（全程与权重无关，只用到 <code>a</code>）`;
    $("#freqCap1").innerHTML =
      `① <b>emb(a)</b> 本体：横轴 = 128 个维度下标（1…128），纵轴 = 该维的数值（<code>W_E[a, c]</code>，纯嵌入矩阵）。` +
      `底部<b>色带</b>标出每一维的<b>主导频率</b>（对每一列沿 token 轴做 DFT，振幅 <code>√(A<sub>k</sub>[c]²+B<sub>k</sub>[c]²)</code> 最大的那个 k），` +
      `箭头指向该频率振幅最大的维度`;
    $("#edDftNote").innerHTML =
      `<b>怎么算出来的：</b>把 <code>one-hot(a)</code> 代进 DFT 定义式 <code>ĉ<sub>k</sub> = Σ<sub>n</sub> x[n]·e<sup>−2πikn/${p}</sup></code>，` +
      `因为 x[n] 只在 n = a 处非零，求和只剩一项：<b>ĉ<sub>k</sub> = e<sup>−2πika/${p}</sup></b>。` +
      `取模得 <b>|ĉ<sub>k</sub>| = 1</b>（对所有 k 都一样），取角得 <b>∠ĉ<sub>k</sub> = −2πka/${p}</b>（随 k <b>线性</b>）。` +
      `<br><b>换成实数记法</b>（直流 + ${K} 对 cos/sin）：直流 <code>c₀ = 1/${p} = ${(1 / p).toFixed(5)}</code>，` +
      `每个频率的振幅都恰好 <code>= 2/${p} = ${(2 / p).toFixed(5)}</code>，相位 <code>φ<sub>k</sub> = 2πka/${p}</code>。` +
      `<br><b>三个纯 DFT 的事实：</b>` +
      `<br>1️⃣ <b>为什么需要 ${K} 个频率</b>：${p} 个实数进 → 1 个直流 + ${K} 对 (cos,sin) 系数出来，` +
      `因为实信号的 DFT 是共轭对称的，k > ${K} 的部分完全重复 k < ${K} 的信息。` +
      `信息量 1 + 2×${K} = <b>${2 * K + 1} = ${p}</b>，一个不多一个不少。` +
      `<br>2️⃣ <b>幅度谱完全与 a 无关</b>：不管 a 是几，幅度都恒为 2/${p} —— 换 token 只会把相位整体平移，幅度谱一个像素都不变。` +
      `所以「token 是几」这个信息<b>只存在于相位里</b>。` +
      `<br>3️⃣ <b>逆 DFT 回得去（右上图）</b>：把这 ${2 * K + 1} 个系数代进逆变换定义式，` +
      `<code>x[n] = c₀ + Σ<sub>k</sub>[a<sub>k</sub>·cos(2πkn/${p}) + b<sub>k</sub>·sin(2πkn/${p})]</code>，` +
      `逐点算出 ${p} 个值 —— <b>精确回到 one-hot(a)</b>（与左上图逐点重合）。` +
      `道理是：在 n = a 处这 ${K} 条波<b>全部同相 → 相长</b>（加起来 = 1）；其它位置彼此错开 → <b>相消</b>（加起来 = 0）。` +
      `右上图里那条浅橙色曲线是「只保留前 16 个频率」的逆变换，峰只有 ${((1 + 32) / p).toFixed(3)} —— ` +
      `保留的频率越多，峰越高越窄（因为 n = a 处每多一个频率就多贡献 2/${p}）。`;

    $("#freqCap1b").innerHTML =
      `② 把 emb(a) 拆开来看：它装着 <b>${K} 个频率</b>。⚠️ <b>不是「第 c 维 = 第 k 个频率」</b>——` +
      `频率是<b>沿 token 轴</b>的周期（横轴 = token n = 0…${p - 1}），而且每个频率 k 都是一对<b>铺满全部 ${d} 维</b>的向量` +
      `<code>(A<sub>k</sub>, B<sub>k</sub>)</code>，所以每一维里都同时混着 ${K} 个频率。` +
      `下图 = 这 ${K} 个频率沿 token 轴各自的样子（一行一条波；右列 R = 该频率在 emb(a) 里那一份的<b>长度</b>）`;
    $("#freqCap3").innerHTML =
      `③ 把这些频率一条条<b>叠加回去</b>（逆变换）：前 4 个 → 前 16 个 → 全部 ${K} 个，峰高始终是 1，旁瓣越来越小，最后正好是 one-hot(a)`;
    $("#edWaveNote").innerHTML =
      `① 这就是 <b>emb(a = ${a})</b> 这个 128 维向量本体：${d} 个数看上去毫无规律，` +
      `但它其实是<b>频域</b>里的东西——把它拆开正好是 <b>${K} 个频率</b>的「cos / sin 对」（下面 ② 逐条画出来）。` +
      `<br>也就是说：one-hot 在时域里是一根尖峰，搬到频域后变成 ${K} 个频率的相位组合；这一步就是 DFT。` +
      `<br><b>⚠️ 「装着 ${K} 个频率」不是在「128 个抽屉」里放频率。</b>` +
      `频率是<b>沿 token 轴</b>（横轴 n = 0…${p - 1}，共 ${p} 个 token）的周期，` +
      `而 128 个维度是<b>纵轴</b>——W_E 是一张 ${p}×${d} 的表：` +
      `<b>取一行</b>（固定 a）= emb(a) 的 ${d} 个数，看不出周期；` +
      `<b>取一列</b>（固定 c）= 沿 token 的 ${p} 个数，才是周期曲线。` +
      `emb(a) 本质上是「这 ${d} 条曲线在横坐标 n = a 处同时取一次样」得到的 ${d} 个纵坐标。` +
      `所以没有任何一维「属于」某个频率：实测 dim 1 的主导频率是 k=20（不是 k=1），而全部 128 维里主导频率为 k=1 的有 <b>0</b> 个。` +
      `<br><b>色带（关键频率在哪几维）：</b>对每一维 c，把它的 128 个「振幅」<code>√(A<sub>k</sub>[c]²+B<sub>k</sub>[c]²)</code> 比一遍，最大的那个 k 就是该维的<b>主导频率</b>。` +
      `实测分布：${dv.ranked.slice(0, 3).map(([k, n]) => `<b>k=${k}</b> 占 ${n} 维`).join("、")}，` +
      `加起来就是 ${dv.covered} / ${d} 维；振幅最大的维分别是 ` +
      `${dv.marks.map((mk) => `<b>dim ${mk.c + 1}</b> → k=${mk.k}`).join("、")}。` +
      `<br>但要注意：这是「<b>主导</b>」不是「独占」——单个维度里主导频率只占中位数 <b>${(100 * dv.medShare).toFixed(0)}%</b> 的能量，` +
      `前 6 个频率合计才占到中位数 <b>${(100 * dv.medTop6).toFixed(0)}%</b>。` +
      `也就是说<b>没有「第 c 维 = 第几个频率」的对应关系</b>（128 维的顺序是 SGD 自己定的），` +
      `色带只是把「这一维主要受哪个频率支配」标出来，真正的频率成分要看 ②。`;
    $("#edSplitNote").innerHTML =
      `② <b>先分清轴</b>：频率在 <b>token 轴</b>上（横轴 n = 0…${p - 1}），不在 128 个维度上。` +
      `每个频率 k 沿 token 轴的样子是 <code>cos(2πk(n−a)/${p})</code>，也就是「${p} 步里转 k 圈」的波。` +
      `<b>关键：这 ${K} 条波的相位全部由 emb(a) 决定，它们在 n = a 处严格同相</b>（绿虚线）——` +
      `所以单看每条波都有很多峰（周期 ${p}/k，谁也定不了 a），但一起加就会在 a 处堆出一个尖峰（见 ③）。` +
      `<br>右列 R = 该频率在 emb(a) 里那一份的<b>长度</b>：<code>R<sub>k</sub> = ‖cos θ<sub>k</sub>·A<sub>k</sub> + sin θ<sub>k</sub>·B<sub>k</sub>‖</code>（θ<sub>k</sub> = 2πka/${p}）。` +
      `A<sub>k</sub>、B<sub>k</sub> 是<b>长度 128 的方向向量</b>（把每一列沿 token 轴做 DFT 得到），所以 R<sub>k</sub> 用的是<b>全部 128 维</b>的平方和，` +
      `<b>不是</b>「第 k 维的取值」——128 维的顺序是 SGD 自己定的（与理想布局余弦相似度只有 0.025），第 1 维和 k=1 没有任何对应关系。` +
      `实测最强的 <b>k = ${a1}</b>（${amps[a1 - 1].toFixed(3)}）和 <b>k = ${a2}</b>（${amps[a2 - 1].toFixed(3)}），` +
      `第 3 名的 k = ${a3} 只有 ${amps[a3 - 1].toFixed(3)}——顿悟后模型实际只用到少数几个频率（稀疏傅里叶电路），` +
      `但它们的相位关系仍然是严格的正弦，所以照样能精确复原。`;
    $("#edPartsNote").innerHTML =
      `③ 理想版逆变换：J 个频率的和 <code>Σ<sub>k≤J</sub>cos(2πk(n−a)/${p})</code> 除以 <code>J+0.5</code> 就是上图曲线（这样峰高恒为 1）。` +
      `J 越大旁瓣越小；<b>全部 ${K} 个用上时</b>，和式在 n = a 处 = ${K}、其它位置 = −1/2，` +
      `归一化后<b>精确等于 one-hot(a)</b>（灰柱）。` +
      `<br>⚠️ <b>这张图与权重无关</b>：它只用到 <code>a</code>，不读 <code>W_E</code>（换 seed / epoch 它一个像素都不变）。` +
      `它证明的是「1 + 2×${K} = ${2 * K + 1} = P 个频率系数是完备的」——所以用频率表示 token 不会丢信息。` +
      `真正<b>从 emb(a) 出发</b>的逆变换见下面 ④。` +
      `<br>自检（这是另一件事）：用这 ${K} 个频率与它们的相位重建 emb(a)，相对误差 <b>${errPct.toExponential(1)}%</b>` +
      `（机器精度，说明「emb(a) = 这 ${K} 个频率」不是比喻，是精确分解）。`;
    $("#freqCap4").innerHTML =
      `④ <b>真实逆变换</b>：不假设任何东西，直接拿 ① 里那 ${d} 个数算出整条曲线 —— 红线与灰柱（真值 one-hot）应当完全重合`;
    $("#edInvNote").innerHTML =
      `<b>两步，全部只吃 emb(a) 的 ${d} 个数：</b>` +
      `<br>① <b>解系数</b>：<code>x = R⁻¹·(Qᵀ·w)</code>（w = emb(a)，128 个数 → ${2 * K + 1} 个频率系数）。` +
      `其中 Q、R 是基矩阵 [DC, A₁, B₁, …, A<sub>${K}</sub>, B<sub>${K}</sub>] 的 QR 分解——` +
      `它由<b>整张 W_E</b> 算出来。实测解出的 <code>x[0]</code> 与 <code>cos/sin θ<sub>k</sub></code> 对真值的最大偏差 = ` +
      `<b>${eCoeff.toExponential(1)}</b>（≈0，说明系数被完整还原了）。` +
      `<br>② <b>读回 token 轴</b>：<code>one-hot(n) = (Σ<sub>j</sub> x<sub>j</sub>·P[n,j] − 0.5)/(${K}+0.5)</code>，` +
      `其中 <code>P[n,j]</code> 就是「token n 自己的相位行」<code>[1, cos(2πkn/${p}), sin(2πkn/${p}), …]</code>` +
      `——所以第 ② 步本质上就是<b>把两个相位行做点积</b>（只有 n = a 时同相 → 相长）。这和 attention 的 QKᵀ 是同一件事。` +
      `<br><b>结果：</b>峰值 <b>${peak.toFixed(6)}</b> 落在 n = ${a}（正确 ✅），与真值 one-hot 的最大偏差 <b>${eHot.toExponential(1)}</b>` +
      `（不为 0 是因为基矩阵的条件数高达 <b>1.18×10⁵</b>——128 维里塞了 ${2 * K + 1} 个方向且彼此不正交，放大机器精度）。` +
      `<br><b>橙线 = 只用最强的 ${J6} 个频率（k = ${top.slice(0, J6).join(", ")}）</b>：` +
      `峰还在 n = ${jArg}（argmax 依然正确 ✅，因为相位是一致的），但旁边鼓起了 <b>${jSecond.toFixed(3)}</b> 的旁瓣 ——` +
      `峰高 ${jPeak.toFixed(3)} / 次高 ${jSecond.toFixed(3)}。` +
      `<b>相位对 ⇒ 一定认得出是哪个 token；频率少 ⇒ 选择性差</b>。` +
      `<br>⚠️ 注意：这个"逆矩阵"是<b>我们</b>用整张表算出来的，<b>不是模型自带的</b>。` +
      `模型最后那步读出用的是学出来的 <code>unembed</code>（见 LOGITS 视图），它和这个精确逆矩阵的余弦相似度只有 0.095 —— 几乎无关。`;
    $("#freqResult").hidden = false;
  }

  // ===================================================================
  // 频率视图：沿数据流逐级看同一件事 —— 某一维是不是少数几个 sin/cos 波
  //   EMBED → ATTENTION → MLP → LOGITS，正好是傅里叶乘法电路的全链路
  //   （读出权重 W_U 的视图已下线：它的“输出”就是 logits，想看权重看下面热图区）
  // ===================================================================
  const FREQ_SRC = {
    embed: {
      name: "EMBED 频率", icon: "🎼", btn: "embedBtn", stage: "embed",
      selLabel: "token a", selKind: "token", varName: "a",
      xAxis: "token a", qty: "emb(a)",
      cap1: "① emb(a) 本体：横轴 = 128 个维度下标（1…128），纵轴 = 该维的数值",
      where: "论文 3.1 第一句 / 图 3（左）",
      why: "<b>词嵌入把 token a 映射到 sin(w<sub>k</sub>a)、cos(w<sub>k</sub>a)</b>：这个 128 维向量不是一张任意查表，而是少数几个 sin/cos 波的固定组合。",
      hint: "三级递进：① 先看 <b>emb(a)</b> 这个 128 维向量本体；② 它其实装着 <b>56 个频率</b>——逐条画出这 56 条波长什么样（相位全由 emb(a) 决定，在 n = a 处同相）；" +
            "③ 把这 56 条波<b>叠加回去</b>（逆变换），波峰唯一落在 n = a，正好复原出 one-hot(a)。" +
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
            "logits 就是残差流经 W_U 读出得到的结果（结构图上不再单画 UNEMBED 权重块，它的作用已体现在这里）。",
      part: "logitsCell",
    },
  };

  /** 取 src 指定的那条长度 p 的曲线（sel = 维度下标或输出类）。
   *  注：EMBED 已改为独立视图（见 runEmbedView），不再走这里。 */
  function freqSignal(src, sel, p, d) {
    const y = new Float64Array(p);
    if (src === "logits") {
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

  /** 频率视图主流程：src ∈ embed / attn / mlp / logits
   *  resync = true 表示由按钮点开（EMBED 会把 token 同步成结构图上的 a）；
   *  换 epoch 引起的自动重算不带这个标志，以免把用户选好的 token 冲掉。 */
  async function runFreqView(src, resync = false) {
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
    if (!state.mats) { alert("权重还没载入成功（看工具栏右侧的状态），切换一下 Seed 或拖动 epoch 重试即可"); return; }
    btn.disabled = true; btn.innerHTML = "分析中…";
    try {
      const p = state.cfg.p, d = state.cfg.d_model;
      // 选择器：dim = 残差流维度，class = 输出类，token = 词表里的 token（EMBED 视图用）
      const sel = $("#freqSel");
      const kind = spec.selKind === "class" ? "class" : spec.selKind === "token" ? "token" : "dim";
      const switched = sel.dataset.src !== src;
      $("#freqSelLabel").textContent = spec.selLabel;
      $("#freqTitle").textContent = `${spec.icon} ${spec.name}`;
      if (switched) {
        sel.innerHTML = "";
        const n = kind === "dim" ? d : p;
        const pre = kind === "class" ? "类 " : kind === "token" ? "token " : "dim ";
        for (let i = 0; i < n; i++) sel.add(new Option(`${pre}${i}`, String(i)));
        sel.dataset.src = src;
      }
      if (kind === "token") {
        // 默认跟随结构图上方那个样本的 a（跑过测试就用测试样本，否则用结构图的默认值）
        if (switched || resync) sel.value = String(state.sample ? state.sample.a : defaultSampleAB(p).a);
      } else if (switched) {
        if (state.freqPick[src] == null) state.freqPick[src] = pickFreqIndex(src, p, d);
        sel.value = String(state.freqPick[src]);
      }
      const topSel = $("#freqTop");
      if (topSel.options.length === 0) {
        for (let n = 1; n <= 8; n++) topSel.add(new Option(String(n), String(n)));
      }
      // 换视图时把 N 复位到该级的默认值（EMBED 不用 N，它的控件会被隐藏）
      if (switched) topSel.value = src === "embed" ? "2" : "3";
      const c = Number(sel.value), N = Number(topSel.value);

      // 维度 / 输出类选择器已经就绪，让结构图上的高亮再同步一次
      // （LOGITS 点亮的格子要跟 “输出类” 下拉框一致；setPanelInfo 调用时下拉框还没建好）
      highlightArch(spec.stage);

      // EMBED 走重构后的三级视图（① emb(a) 本体 ② 56 个频率 ③ 叠加回 one-hot(a)）
      if (src === "embed") { runEmbedView(c); return; }
      // 其它三级视图：恢复被 EMBED 藏起来的那一块，并隐藏 EMBED 专属的 ②
      for (const id of ["#freqCap2", "#cEmbedSpec", "#edSpecNote"]) $(id).hidden = false;
      const nSel2 = $("#freqTop");
      if (nSel2 && nSel2.closest(".kv")) nSel2.closest(".kv").style.display = "";
      $("#embedSplitWrap").hidden = true;
      $("#embedDftWrap").hidden = true;

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
      state.index = await fetchJSON("/viz-data/index.json");    } catch (e) {
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
    const tb = $("#testBtn");
    if (tb) tb.addEventListener("click", runTest);
    // ⓪ 卡片的视角切换：效果（准确率曲线）/ 参数（W_E 频率谱）
    for (const b of document.querySelectorAll("#seedModes button")) {
      b.onclick = () => setSeedMode(b.dataset.mode);
    }
    setSeedMode("curve");
    // 五个“频率”按钮：同一套视图，只是数据来源不同（沿数据流 EMBED → … → LOGITS）
    for (const key of Object.keys(FREQ_SRC)) {
      const b = $(`#${FREQ_SRC[key].btn}`);
      if (b) b.addEventListener("click", () => runFreqView(key, true));
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
    // 卡片右上角的全屏按钮（⛶ / ✕）
    for (const b of document.querySelectorAll("[data-fs-btn]")) {
      b.addEventListener("click", () => {
        const card = b.closest("[data-fs]");
        if (card) toggleCardFullscreen(card);
      });
    }
    syncFsBtns();
    // 浏览器全屏状态变了（含用户按 Esc / F11）：同步图标并按新尺寸重画
    document.addEventListener("fullscreenchange", () => {
      syncFsBtns();
      for (const card of document.querySelectorAll("[data-fs]")) {
        const fn = FS_REDRAW[card.dataset.fs];
        if (fn) fn();
      }
    });
    // 退化（class）全屏：Esc 退出并重画
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      for (const card of document.querySelectorAll("[data-fs].fs-fallback")) {
        card.classList.remove("fs-fallback");
        const fn = FS_REDRAW[card.dataset.fs];
        if (fn) fn();
      }
      syncFsBtns();
    });
    // 窗口尺寸变化时，正在全屏的卡片要重新量一下空间并重画
    window.addEventListener("resize", () => {
      for (const card of document.querySelectorAll("[data-fs]")) {
        if (!cardIsFull(card.dataset.fs)) continue;
        const fn = FS_REDRAW[card.dataset.fs];
        if (fn) fn();
      }
    });
    // 频率面板右上角的「i」：用 iframe 弹层打开「时域 → 频域」基础教程
    const infoModal = $("#infoModal"), infoFrame = $("#infoFrame");
    if (infoModal && infoFrame) {
      const openInfo = () => {
        const src = infoFrame.getAttribute("src");
        if (!src || src === "about:blank") infoFrame.src = "/fourier-basics.html";   // 首次点开才加载
        infoModal.hidden = false;
        document.body.style.overflow = "hidden";
      };
      const closeInfo = () => {
        infoModal.hidden = true;
        document.body.style.overflow = "";
      };
      const ib = $("#freqInfoBtn");
      if (ib) ib.addEventListener("click", openInfo);
      const ic = $("#infoClose");
      if (ic) ic.addEventListener("click", closeInfo);
      infoModal.addEventListener("click", (e) => { if (e.target === infoModal) closeInfo(); });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !infoModal.hidden) closeInfo();
      });
    }
    // 模加法表：鼠标悬停显示这一对 (a,b)
    const addCv = $("#cAdd");
    if (addCv) {
      addCv.addEventListener("mousemove", addTableTip);
      addCv.addEventListener("mouseleave", () => {
        const t = $("#addTip");
        if (t) t.hidden = true;
      });
    }
    onRun();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
