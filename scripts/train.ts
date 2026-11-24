/**
 * Budget-aware training CLI.
 *
 *   npm run train -- --budget-min 150 --run release
 *   npm run train -- --budget-min 2 --run smoke --synthetic --preset small --workers 2
 *
 * Reads data/train.bin + data/val.bin (raw uint16 token ids). Writes models/<run>/{config.json,
 * weights.bin (best EMA weights on val loss), last.bin, log.jsonl, summary.json}.
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { countParams, makeConfig, trainFlopsPerToken, type PresetName } from "../src/model/index.js";
import { loadTokenShard, maxToken, readVocabFromMeta, syntheticTokens } from "../src/train/data.js";
import { defaultWorkers, train, type LogRecord } from "../src/train/trainer.js";

const { values: a } = parseArgs({
  options: {
    "budget-min": { type: "string" },
    minutes: { type: "string" },
    run: { type: "string", default: "dev" },
    preset: { type: "string" },
    vocab: { type: "string" },
    ctx: { type: "string" },
    d: { type: "string" },
    layers: { type: "string" },
    heads: { type: "string" },
    mlp: { type: "string" },
    workers: { type: "string" },
    batch: { type: "string", default: "8" },
    lr: { type: "string", default: "3e-3" },
    "min-lr-frac": { type: "string", default: "0.1" },
    warmup: { type: "string", default: "200" },
    "warmup-rho": { type: "string", default: "0.03" },
    wd: { type: "string", default: "0.1" },
    clip: { type: "string", default: "1.0" },
    ema: { type: "string", default: "0.999" },
    "eval-every-min": { type: "string", default: "10" },
    "eval-tokens": { type: "string", default: "131072" },
    "log-every-sec": { type: "string", default: "15" },
    "max-steps": { type: "string" },
    "no-ctx-curriculum": { type: "boolean", default: false },
    "ctx-switch-rho": { type: "string", default: "0.3" },
    seed: { type: "string", default: "1" },
    data: { type: "string", default: "data" },
    models: { type: "string", default: "models" },
    synthetic: { type: "boolean", default: false },
  },
  allowPositionals: false,
});

const num = (v: string | undefined): number | undefined => (v === undefined ? undefined : Number(v));
const budgetMin = num(a["budget-min"] ?? a.minutes) ?? 150;

// Preset: explicit flag, else the one chosen by the benchmark, else "base".
let preset: PresetName = "base";
const benchPath = join("results", "bench.json");
let benchChoice: string | undefined;
let benchWorkers: number | undefined;
if (existsSync(benchPath)) {
  try {
    const b = JSON.parse(readFileSync(benchPath, "utf8")) as { choice?: { preset?: string }; recommendedWorkers?: number };
    benchChoice = b.choice?.preset;
    benchWorkers = b.recommendedWorkers;
  } catch {
    benchChoice = undefined;
  }
}
if (a.preset) preset = a.preset as PresetName;
else if (benchChoice === "small" || benchChoice === "base" || benchChoice === "large") preset = benchChoice;

// Data.
let trainTokens: Uint16Array, valTokens: Uint16Array;
let vocab = num(a.vocab) ?? readVocabFromMeta(a.data!) ?? 2048;
if (a.synthetic) {
  const all = syntheticTokens(2_200_000, vocab, Number(a.seed));
  trainTokens = all.subarray(0, 2_000_000);
  valTokens = all.subarray(2_000_000);
} else {
  const tp = join(a.data!, "train.bin"), vp = join(a.data!, "val.bin");
  if (!existsSync(tp) || !existsSync(vp)) {
    console.error(`missing ${tp} / ${vp}. Build the corpus + tokenizer first, or pass --synthetic for a smoke run.`);
    process.exit(2);
  }
  trainTokens = loadTokenShard(tp);
  valTokens = loadTokenShard(vp);
  const mx = Math.max(maxToken(trainTokens), maxToken(valTokens));
  if (mx >= vocab) throw new Error(`token id ${mx} >= vocab ${vocab}; pass --vocab`);
}

const cfg = makeConfig(preset, vocab, {
  ...(a.ctx ? { ctx: Number(a.ctx) } : {}),
  ...(a.d ? { dModel: Number(a.d) } : {}),
  ...(a.layers ? { nLayers: Number(a.layers) } : {}),
  ...(a.heads ? { nHeads: Number(a.heads) } : {}),
  ...(a.mlp ? { dMlp: Number(a.mlp) } : a.d ? { dMlp: 4 * Number(a.d) } : {}),
});
const workers = num(a.workers) ?? benchWorkers ?? defaultWorkers();
const batch = Number(a.batch);

console.log(
  `[train] run=${a.run} preset=${preset}${benchChoice && !a.preset ? " (from results/bench.json)" : ""} params=${countParams(cfg).toLocaleString()} ` +
    `cfg=${JSON.stringify(cfg)}`,
);
console.log(
  `[train] budget=${budgetMin} min workers=${workers} batch/worker=${batch} tokens/step=${Math.max(1, workers) * batch * cfg.ctx} ` +
    `train tokens=${trainTokens.length.toLocaleString()} val tokens=${valTokens.length.toLocaleString()}${a.synthetic ? " (synthetic)" : ""}`,
);

const fmt = (r: LogRecord): string => {
  const base = `step ${String(r.step).padStart(6)} rho ${r.rho.toFixed(3)} t ${(r.t / 60).toFixed(1)}m tok ${(r.tokens / 1e6).toFixed(2)}M ` +
    `lr ${r.lr.toExponential(2)} T ${r.ctx} loss ${r.loss.toFixed(4)} gnorm ${r.gnorm.toFixed(2)} ${Math.round(r.tok_s)} tok/s`;
  return r.val_loss !== undefined
    ? `${base} | val ${r.val_loss.toFixed(4)}${r.val_loss_raw !== undefined ? ` (raw ${r.val_loss_raw.toFixed(4)})` : ""}${r.best ? " *best" : ""}`
    : base;
};

const summary = await train(
  cfg,
  { train: trainTokens, val: valTokens },
  {
    run: a.run!,
    modelsDir: a.models!,
    budgetMin,
    ...(a["max-steps"] ? { maxSteps: Number(a["max-steps"]) } : {}),
    workers,
    batchPerWorker: batch,
    lr: { peakLr: Number(a.lr), minLrFrac: Number(a["min-lr-frac"]), warmupSteps: Number(a.warmup), warmupRho: Number(a["warmup-rho"]) },
    ctx: { enabled: !a["no-ctx-curriculum"], startCtx: Math.max(16, cfg.ctx / 2), fullCtx: cfg.ctx, switchRho: Number(a["ctx-switch-rho"]) },
    weightDecay: Number(a.wd),
    beta1: 0.9,
    beta2: 0.95,
    clip: Number(a.clip),
    emaDecay: Number(a.ema),
    evalEveryMin: Number(a["eval-every-min"]),
    evalTokens: Number(a["eval-tokens"]),
    logEverySec: Number(a["log-every-sec"]),
    seed: Number(a.seed),
    meta: {
      preset,
      data: a.synthetic ? "synthetic" : { dir: a.data, train: trainTokens.length, val: valTokens.length },
      flopsPerToken: trainFlopsPerToken(cfg),
    },
  },
  (r) => console.log(fmt(r)),
);

console.log(`[train] done: ${JSON.stringify(summary)}`);
