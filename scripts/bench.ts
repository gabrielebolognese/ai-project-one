/**
 * Measure before choosing (SPEC §5.4, TinyMind's bench_gemm.py lesson):
 *   1. GEMM throughput of our kernels at the model's real layer shapes: naive JS, register-tiled
 *      JS and the hand-assembled WASM SIMD kernel, single thread.
 *   2. End-to-end data-parallel training throughput (tokens/s, the real trainer step: workers'
 *      forward+backward, gradient all-reduce, clipping, AdamW, EMA) for 3 candidate model sizes,
 *      plus worker scaling 1/2/4/8/16 for the base size.
 *   3. Pick the largest candidate that sees >= 10 tokens/parameter in the release budget.
 *
 *   npm run bench                 (about 5 minutes; writes results/bench.json)
 *   npm run bench -- --quick      (shorter timings, for a smoke check)
 */
import { cpus, totalmem } from "node:os";
import { mkdirSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { AdamW, Rng, clipGradNorm, gemmNTCore, gemmNaive, gemmNTWasm, wasmGemmAvailable } from "../src/engine/index.js";
import { PRESETS, Transformer, countParams, makeConfig, trainFlopsPerToken, type ModelConfig, type PresetName } from "../src/model/index.js";
import { sampleOffsets, syntheticTokens } from "../src/train/data.js";
import { WorkerPool, averageGrads } from "../src/train/executor.js";

const { values: args } = parseArgs({
  options: {
    quick: { type: "boolean", default: false },
    "budget-min": { type: "string", default: "150" },
    vocab: { type: "string", default: "2048" },
    batch: { type: "string", default: "8" },
    workers: { type: "string", default: "8" },
    out: { type: "string", default: "results/bench.json" },
  },
});
const QUICK = args.quick!;
const BUDGET_MIN = Number(args["budget-min"]);
const VOCAB = Number(args.vocab);
const BATCH = Number(args.batch);
const WORKERS = Number(args.workers);
const MIN_TOKENS_PER_PARAM = 10;
/** Fraction of the budget left for optimiser steps after periodic evals and logging. */
const STEP_TIME_FRACTION = 0.97;

const log = (...a: unknown[]): void => console.log(...a);

/**
 * Background CPU load (percent of all logical cores busy) sampled while this process is idle.
 * The owner's laptop runs other jobs; a benchmark taken under load under-reports throughput, so
 * the load is recorded next to every measurement and the run is flagged as contended.
 */
async function backgroundLoad(ms = 2000): Promise<number> {
  const snap = (): { idle: number; total: number } => {
    let idle = 0, total = 0;
    for (const c of cpus()) {
      const t = c.times;
      idle += t.idle;
      total += t.user + t.nice + t.sys + t.irq + t.idle;
    }
    return { idle, total };
  };
  const a = snap();
  await new Promise((r) => setTimeout(r, ms));
  const b = snap();
  const total = b.total - a.total;
  return total > 0 ? +(100 * (1 - (b.idle - a.idle) / total)).toFixed(1) : 0;
}
const CONTENDED_LOAD = 15;

// ------------------------------------------------------------------ 1. kernels

type Kernel = (M: number, N: number, K: number, A: Float32Array, B: Float32Array, C: Float32Array) => void;

function timeKernel(M: number, N: number, K: number, f: Kernel, seconds: number): number {
  const r = new Rng(M * 7 + N * 3 + K);
  const A = new Float32Array(M * K), B = new Float32Array(N * K), C = new Float32Array(M * N);
  for (let i = 0; i < A.length; i++) A[i] = r.next() - 0.5;
  for (let i = 0; i < B.length; i++) B[i] = r.next() - 0.5;
  for (let i = 0; i < 2; i++) f(M, N, K, A, B, C);
  let n = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < seconds * 1000 || n < 3) {
    f(M, N, K, A, B, C);
    n++;
  }
  const dt = (performance.now() - t0) / 1000 / n;
  return (2 * M * N * K) / dt / 1e9;
}

const jsTiled: Kernel = (M, N, K, A, B, C) => gemmNTCore(M, N, K, A, 0, B, 0, C, 0, false);
const wasm: Kernel = (M, N, K, A, B, C) => {
  if (!gemmNTWasm(M, N, K, A, 0, B, 0, C, 0, false)) throw new Error("wasm kernel declined");
};
const naive: Kernel = (M, N, K, A, B, C) => gemmNaive(M, N, K, A, B, C);

const base = makeConfig("base", VOCAB);
const BT = BATCH * base.ctx;
const d = base.dModel, m = base.dMlp, hd = base.dModel / base.nHeads, T = base.ctx;
const shapes: { name: string; M: number; N: number; K: number }[] = [
  { name: "attn.qkv fwd  (BT x d) @ (d x 3d)", M: BT, N: 3 * d, K: d },
  { name: "mlp.fc fwd    (BT x d) @ (d x 4d)", M: BT, N: m, K: d },
  { name: "mlp.proj fwd  (BT x 4d) @ (4d x d)", M: BT, N: d, K: m },
  { name: "head fwd      (BT x d) @ (d x V)", M: BT, N: VOCAB, K: d },
  { name: "mlp.fc dW     (4d x BT) @ (BT x d)", M: m, N: d, K: BT },
  { name: "attn scores   (T x hd) @ (hd x T)", M: T, N: T, K: hd },
  { name: "attn values   (T x T) @ (T x hd)", M: T, N: hd, K: T },
];

const loadBefore = await backgroundLoad();
log(`[bench] CPU: ${cpus()[0]?.model ?? "?"} x${cpus().length} logical, node ${process.version}, WASM SIMD: ${wasmGemmAvailable()}`);
log(`[bench] background CPU load before: ${loadBefore}%${loadBefore > CONTENDED_LOAD ? "  (CONTENDED: other jobs are running; throughput will be under-reported)" : ""}`);
log(`[bench] 1/3 GEMM kernels, single thread, base shapes (B=${BATCH}, T=${T}, d=${d}, V=${VOCAB})`);
const kSec = QUICK ? 0.3 : 1.0;
const kernelRows = shapes.map((s) => {
  const js = timeKernel(s.M, s.N, s.K, jsTiled, kSec);
  const wa = wasmGemmAvailable() ? timeKernel(s.M, s.N, s.K, wasm, kSec) : NaN;
  const row = { ...s, js_gflops: +js.toFixed(2), wasm_gflops: +wa.toFixed(2), speedup: +(wa / js).toFixed(2) };
  log(`  ${s.name.padEnd(38)} M=${String(s.M).padStart(5)} N=${String(s.N).padStart(5)} K=${String(s.K).padStart(5)}  js ${row.js_gflops.toFixed(2).padStart(5)}  wasm ${row.wasm_gflops.toFixed(2).padStart(5)} GFLOPS  (x${row.speedup})`);
  return row;
});
const naiveGflops = +timeKernel(BT, m, d, naive, kSec).toFixed(2);
log(`  naive i-k-j JS reference at mlp.fc shape: ${naiveGflops} GFLOPS`);
const meanSpeedup = kernelRows.reduce((a, r) => a + r.speedup, 0) / kernelRows.length;
const wasmAdopted = wasmGemmAvailable() && meanSpeedup > 1.5;
log(`  mean WASM speedup x${meanSpeedup.toFixed(2)} -> ${wasmAdopted ? "adopt WASM SIMD (> 1.5x rule)" : "keep JS kernel"}`);

// ------------------------------------------------------------------ 2. end-to-end training throughput

interface Throughput {
  workers: number;
  batchPerWorker: number;
  steps: number;
  tok_s: number;
  step_ms: number;
  sustained_gflops: number;
}

async function trainThroughput(cfg: ModelConfig, workers: number, seconds: number): Promise<Throughput> {
  const P = countParams(cfg);
  const params = new Float32Array(new SharedArrayBuffer(P * 4));
  const model = new Transformer(cfg, { data: params });
  model.init(1);
  const tokens = syntheticTokens(400_000, cfg.vocabSize, 2);
  const val = tokens.subarray(0, 10_000);
  const pool = await WorkerPool.create({ cfg, params, train: tokens, val }, workers);
  const opt = new AdamW(P, model.store.decayMask());
  const ema = Float32Array.from(params);
  const grad = model.store.grad;
  const rng = new Rng(3);
  const step = async (): Promise<void> => {
    const offs = Array.from({ length: workers }, () => sampleOffsets(rng, tokens.length, cfg.ctx, BATCH));
    await pool.step(offs, cfg.ctx);
    averageGrads(pool.grads, grad);
    clipGradNorm(grad, 1);
    opt.update(params, grad, 1e-3);
    for (let i = 0; i < P; i++) ema[i] = 0.999 * ema[i] + 0.001 * params[i];
  };
  try {
    for (let i = 0; i < 2; i++) await step(); // warm up JIT + WASM in every worker
    let n = 0;
    const t0 = performance.now();
    while (performance.now() - t0 < seconds * 1000 || n < 3) {
      await step();
      n++;
    }
    const sec = (performance.now() - t0) / 1000;
    const tok = n * workers * BATCH * cfg.ctx;
    return {
      workers, batchPerWorker: BATCH, steps: n, tok_s: +(tok / sec).toFixed(1), step_ms: +((sec / n) * 1000).toFixed(1),
      sustained_gflops: +((tok * trainFlopsPerToken(cfg)) / sec / 1e9).toFixed(2),
    };
  } finally {
    await pool.close();
  }
}

const eSec = QUICK ? 4 : 20;
log(`[bench] 2/3 end-to-end training throughput (${WORKERS} workers x ${BATCH} seqs x ${T} tokens per step)`);
const presetNames: PresetName[] = ["small", "base", "large"];
const budgetSec = BUDGET_MIN * 60 * STEP_TIME_FRACTION;
const candidates = [];
for (const p of presetNames) {
  const cfg = makeConfig(p, VOCAB);
  const thr = await trainThroughput(cfg, WORKERS, eSec);
  const params = countParams(cfg);
  const tokensInBudget = Math.round(thr.tok_s * budgetSec);
  const row = {
    preset: p, cfg, params, flopsPerToken: trainFlopsPerToken(cfg), ...thr,
    tokens_in_budget: tokensInBudget, tokens_per_param: +(tokensInBudget / params).toFixed(1),
  };
  candidates.push(row);
  log(`  ${p.padEnd(6)} ${params.toLocaleString().padStart(10)} params  ${String(thr.tok_s).padStart(8)} tok/s  ${String(thr.step_ms).padStart(7)} ms/step  ` +
    `${thr.sustained_gflops} GFLOPS  -> ${(tokensInBudget / 1e6).toFixed(1)}M tokens in ${BUDGET_MIN} min = ${row.tokens_per_param} tokens/param`);
}

log(`[bench]     worker scaling, base preset`);
const scaling: Throughput[] = [];
for (const w of QUICK ? [1, WORKERS] : [1, 2, 4, 8, 16]) {
  const thr = await trainThroughput(base, w, QUICK ? 3 : 12);
  scaling.push(thr);
  log(`  workers ${String(w).padStart(2)}: ${String(thr.tok_s).padStart(8)} tok/s  (${thr.sustained_gflops} GFLOPS, ${thr.step_ms} ms/step)`);
}

// ------------------------------------------------------------------ 3. decision

const loadAfter = await backgroundLoad();
const contended = loadBefore > CONTENDED_LOAD || loadAfter > CONTENDED_LOAD;
log(`[bench] background CPU load after: ${loadAfter}%${contended ? "  -> run flagged as contended" : ""}`);

const ok = candidates.filter((c) => c.tokens_per_param >= MIN_TOKENS_PER_PARAM);
const chosen = ok.length ? ok[ok.length - 1] : candidates[0];
const bestScaling = scaling.reduce((a, b) => (b.tok_s > a.tok_s ? b : a));
const reason = ok.length
  ? `largest candidate with >= ${MIN_TOKENS_PER_PARAM} tokens/param in ${BUDGET_MIN} min: ${chosen.preset} sees ${(chosen.tokens_in_budget / 1e6).toFixed(1)}M tokens ` +
    `(${chosen.tokens_per_param} tokens/param) at ${chosen.tok_s} tok/s with ${WORKERS} workers`
  : `no candidate reaches ${MIN_TOKENS_PER_PARAM} tokens/param; fall back to the smallest (${chosen.preset})`;
log(`[bench] 3/3 choice: ${chosen.preset} -- ${reason}`);
log(`        best worker count measured: ${bestScaling.workers} (${bestScaling.tok_s} tok/s)`);

const out = {
  date: new Date().toISOString(),
  machine: { cpu: cpus()[0]?.model ?? "unknown", logicalCores: cpus().length, totalMemGB: +(totalmem() / 2 ** 30).toFixed(1), node: process.version, v8: process.versions.v8 },
  quick: QUICK,
  backgroundLoad: { beforePct: loadBefore, afterPct: loadAfter, contended, thresholdPct: CONTENDED_LOAD },
  settings: { vocab: VOCAB, batchPerWorker: BATCH, workers: WORKERS, ctx: T, budgetMin: BUDGET_MIN, stepTimeFraction: STEP_TIME_FRACTION, minTokensPerParam: MIN_TOKENS_PER_PARAM },
  kernel: { singleThread: kernelRows, naiveIkjGflops: naiveGflops, meanWasmSpeedup: +meanSpeedup.toFixed(2), wasmAdopted },
  presets: PRESETS,
  candidates,
  scaling,
  recommendedWorkers: bestScaling.workers,
  choice: { preset: chosen.preset, params: chosen.params, cfg: chosen.cfg, tok_s: chosen.tok_s, tokens_in_budget: chosen.tokens_in_budget, tokens_per_param: chosen.tokens_per_param, workers: WORKERS, reason },
};
mkdirSync("results", { recursive: true });
writeFileSync(args.out!, JSON.stringify(out, null, 2) + "\n");
log(`[bench] wrote ${args.out}`);
