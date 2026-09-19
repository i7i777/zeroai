// 临时探针：热图每格到底代表什么（用完即删）
const fs = require("fs");
const P = 113, D = 128;
const KEYS = [5, 10, 15, 17, 31, 48];
const dir = "public/viz-data/mainline_p113_frac0.3_wd1_l1_d128_h4x32_m512_ep25000/seed0";
const buf = fs.readFileSync(dir + "/e025000.bin");
const hlen = new DataView(buf.buffer, buf.byteOffset, buf.byteLength).getUint32(4, true);
const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf.buffer, buf.byteOffset + 8, hlen)));
const f32 = new Float32Array(buf.buffer.slice(buf.byteOffset + 8 + hlen));
const mats = {};
for (const m of header.mats) mats[m.name] = { shape: m.shape, data: f32.slice(m.start, m.start + m.numel) };
const WE = mats.token_embed.data;

// 每列在每个关键频率上的幅度 a_k,c = sqrt(a² + b²)
const amp = KEYS.map(() => new Float64Array(D));
for (let c = 0; c < D; c++) {
  const y = new Float64Array(P);
  for (let n = 0; n < P; n++) y[n] = WE[n * D + c];
  KEYS.forEach((k, j) => {
    let sa = 0, sb = 0;
    for (let n = 0; n < P; n++) { const t = 2 * Math.PI * k * n / P; sa += y[n] * Math.cos(t); sb += y[n] * Math.sin(t); }
    const a = 2 / P * sa, b = 2 / P * sb;
    amp[j][c] = Math.hypot(a, b);
  });
}
const ampMax = Math.max(...amp.flatMap((r) => [...r]));
console.log(`全矩阵最大单频幅度 ampMax = ${ampMax.toFixed(3)}`);
console.log("\n每个频率一行：有多强、覆盖多少维");
KEYS.forEach((k, j) => {
  const arr = [...amp[j]];
  const nz = arr.filter((v) => v > 0.004 * ampMax).length;
  const strong = arr.filter((v) => v > 0.2 * ampMax).length;
  console.log(`  k=${String(k).padStart(2)}: 最大 ${Math.max(...arr).toFixed(3)}  有信号(>0.4%)的维度 ${nz}/128  较强(>20%)的维度 ${strong}`);
});
// 每一维的"总强度"（6 个频率幅度之和），看网络实际用了几维
const colSum = Array.from({ length: D }, (_, c) => KEYS.reduce((s, _, j) => s + amp[j][c], 0));
const live = colSum.filter((v) => v > 0.1 * Math.max(...colSum)).length;
const sorted = colSum.map((v, c) => ({ c, v })).sort((a, b) => b.v - a.v);
console.log(`\n每列强度：最强的 12 列 = ${sorted.slice(0, 12).map((t) => `c${t.c}(${t.v.toFixed(2)})`).join(" ")}`);
console.log(`  强度 > 最大值 10% 的列数：${live} / 128`);
console.log(`  强度 > 最大值 1%  的列数：${colSum.filter((v) => v > 0.01 * Math.max(...colSum)).length} / 128`);
// 逐列：几个频率同时亮？
const multi = colSum.map((_, c) => KEYS.filter((_, j) => amp[j][c] > 0.2 * ampMax).length);
console.log(`\n每列有几个"强频率"(>20% ampMax)：0 个 ${multi.filter((n) => n === 0).length} 列、1 个 ${multi.filter((n) => n === 1).length} 列、2 个 ${multi.filter((n) => n === 2).length} 列、≥3 个 ${multi.filter((n) => n >= 3).length} 列`);
// 举一个具体列
const c0 = 29;
console.log(`\n例：第 c=${c0} 列在 6 个频率上的幅度 = ` + KEYS.map((k, j) => `k${k}:${amp[j][c0].toFixed(3)}`).join("  "));
