/**
 * Budget-aware trainer (TinyMind style): progress rho = elapsed / budget drives LR warmup + cosine
 * decay and the context curriculum, so a 20-minute ablation and a 150-minute release run both end
 * annealed. Keeps an EMA of the weights, evaluates it on fixed validation windows every
 * `evalEveryMin` minutes, saves the best EMA weights to models/<run>/weights.bin and logs every
 * step summary to models/<run>/log.jsonl.
 */
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { cpus } from "node:os";
import { AdamW, Rng, clipGradNorm, writeFloat32, writeJson } from "../engine/index.js";
import { Transformer, countParams, trainFlopsPerToken, saveModel, type ModelConfig } from "../model/index.js";
import { evalOffsets, sampleOffsets } from "./data.js";
import { LocalExecutor, WorkerPool, averageGrads, type Executor } from "./executor.js";
import { ctxAt, emaDecayAt, lrAt, type CtxCurriculum, type LrSchedule } from "./schedule.js";

export interface TrainOptions {
  run: string;
  modelsDir: string;
  budgetMin: number;
  /** Optional hard cap on optimiser steps (smoke tests). */
  maxSteps?: number;
  /** 0 = run in-process on the main thread. */
  workers: number;
  batchPerWorker: number;
  lr: LrSchedule;
  ctx: CtxCurriculum;
  weightDecay: number;
  beta1: number;
  beta2: number;
  clip: number;
  emaDecay: number;
  evalEveryMin: number;
  evalTokens: number;
  logEverySec: number;
  seed: number;
  /** Free-form provenance stored in config.json (data paths, tokenizer, preset name...). */
  meta?: Record<string, unknown>;
}

export interface TrainSummary {
  run: string;
  steps: number;
  tokens: number;
  elapsedSec: number;
  tokensPerSec: number;
  bestValLoss: number;
  bestStep: number;
  finalValLossEma: number;
  finalValLossRaw: number;
  numParams: number;
  sustainedGflops: number;
}

export interface LogRecord {
  t: number;
  rho: number;
  step: number;
  tokens: number;
  lr: number;
  ctx: number;
  loss: number;
  gnorm: number;
  tok_s: number;
  val_loss?: number;
  val_loss_raw?: number;
  best?: boolean;
}

export type LogSink = (r: LogRecord) => void;

export function defaultWorkers(): number {
  // Physical cores on the 5700U = 8 (16 SMT threads); data-parallel JS scales with cores.
  return Math.max(1, Math.min(8, Math.floor(cpus().length / 2)));
}

export async function train(
  cfg: ModelConfig,
  data: { train: Uint16Array; val: Uint16Array },
  opts: TrainOptions,
  onLog?: LogSink,
): Promise<TrainSummary> {
  const runDir = join(opts.modelsDir, opts.run);
  mkdirSync(runDir, { recursive: true });
  const logPath = join(runDir, "log.jsonl");
  writeFileSync(logPath, "");

  const params = new Float32Array(new SharedArrayBuffer(countParams(cfg) * 4));
  const model = new Transformer(cfg, { data: params });
  model.init(opts.seed);
  const P = model.numParams;
  const ema = Float32Array.from(params);
  const backup = new Float32Array(P);
  const bestW = Float32Array.from(ema);
  const grad = model.store.grad;
  const opt = new AdamW(P, model.store.decayMask(), { beta1: opts.beta1, beta2: opts.beta2, weightDecay: opts.weightDecay });

  const execInit = { cfg, params, train: data.train, val: data.val };
  const exec: Executor = opts.workers <= 0 ? new LocalExecutor(execInit) : await WorkerPool.create(execInit, opts.workers);
  const W = exec.replicas;

  const valOffs = evalOffsets(data.val.length, cfg.ctx, opts.evalTokens);
  const rng = new Rng(opts.seed + 1);
  const savedConfig = (extra: Record<string, unknown>): Record<string, unknown> => ({
    train: { ...opts, meta: undefined },
    ...(opts.meta ?? {}),
    ...extra,
  });

  // Write config.json up front so the lab can list tensors while the run is going.
  saveModel(runDir, model, savedConfig({ status: "running" }), { data: ema });

  let stopRequested = false;
  const onSig = (): void => {
    if (!stopRequested) console.log("\n[train] stop requested: finishing with a final eval + checkpoint");
    stopRequested = true;
  };
  process.on("SIGINT", onSig);

  const evalWith = async (weights: Float32Array): Promise<number> => {
    backup.set(params);
    params.set(weights);
    try {
      const r = await exec.evaluate(valOffs, cfg.ctx);
      return r.lossSum / Math.max(1, r.count);
    } finally {
      params.set(backup);
    }
  };

  const budgetSec = opts.budgetMin * 60;
  const t0 = performance.now();
  const elapsed = (): number => (performance.now() - t0) / 1000;
  let step = 0, tokens = 0, tokensAtLastLog = 0, timeAtLastLog = 0;
  let lossAcc = 0, lossN = 0, gnormLast = 0, lastLr = 0, lastT = cfg.ctx;
  let best = Infinity, bestStep = -1;
  let nextEval = opts.evalEveryMin * 60;
  let evalSec = 0;
  let lastValEma = NaN, lastValRaw = NaN;
  let bestInfo: Record<string, number> | null = null;

  const log = (r: LogRecord): void => {
    appendFileSync(logPath, JSON.stringify(r) + "\n");
    onLog?.(r);
  };

  const doEval = async (final: boolean): Promise<void> => {
    const te = performance.now();
    const v = await evalWith(ema);
    let vRaw: number | undefined;
    if (final) vRaw = await evalWith(params);
    const isBest = v < best;
    lastValEma = v;
    if (vRaw !== undefined) lastValRaw = vRaw;
    if (isBest) {
      best = v;
      bestStep = step;
      bestW.set(ema);
      bestInfo = { step, tokens, valLoss: v, t: elapsed() };
      saveModel(runDir, model, savedConfig({ status: "running", best: bestInfo }), { data: bestW });
    }
    evalSec += (performance.now() - te) / 1000;
    const t = elapsed();
    log({
      t, rho: Math.min(1, t / budgetSec), step, tokens, lr: lastLr, ctx: lastT, loss: lossN ? lossAcc / lossN : NaN,
      gnorm: gnormLast, tok_s: tokens / Math.max(1e-9, t), val_loss: v, ...(vRaw !== undefined ? { val_loss_raw: vRaw } : {}), best: isBest,
    });
  };

  try {
    while (!stopRequested) {
      const t = elapsed();
      const rho = t / budgetSec;
      if (rho >= 1 || (opts.maxSteps !== undefined && step >= opts.maxSteps)) break;
      const T = ctxAt(opts.ctx, rho);
      const lr = lrAt(opts.lr, step, rho);
      const offsets = Array.from({ length: W }, () => sampleOffsets(rng, data.train.length, T, opts.batchPerWorker));
      const { losses } = await exec.step(offsets, T);
      averageGrads(exec.grads, grad);
      gnormLast = clipGradNorm(grad, opts.clip);
      opt.update(params, grad, lr);
      const d = emaDecayAt(opts.emaDecay, step);
      for (let i = 0; i < P; i++) ema[i] = d * ema[i] + (1 - d) * params[i];
      step++;
      tokens += W * opts.batchPerWorker * T;
      for (const l of losses) { lossAcc += l; lossN++; }
      lastLr = lr;
      lastT = T;
      if (!Number.isFinite(losses[0])) throw new Error(`non-finite loss at step ${step}`);

      const now = elapsed();
      if (now - timeAtLastLog >= opts.logEverySec || step === 1) {
        log({
          t: now, rho: Math.min(1, now / budgetSec), step, tokens, lr, ctx: T, loss: lossAcc / lossN, gnorm: gnormLast,
          tok_s: (tokens - tokensAtLastLog) / Math.max(1e-9, now - timeAtLastLog),
        });
        lossAcc = 0; lossN = 0; tokensAtLastLog = tokens; timeAtLastLog = now;
      }
      if (now >= nextEval) {
        await doEval(false);
        nextEval += opts.evalEveryMin * 60;
      }
    }
    await doEval(true);
  } finally {
    process.off("SIGINT", onSig);
    await exec.close();
  }

  writeFloat32(join(runDir, "last.bin"), params);
  const el = elapsed();
  const trainSec = Math.max(1e-9, el - evalSec);
  const lastLine = await import("node:fs").then((fs) => fs.readFileSync(logPath, "utf8").trim().split("\n").pop() ?? "{}");
  const final = JSON.parse(lastLine) as LogRecord;
  const summary: TrainSummary = {
    run: opts.run,
    steps: step,
    tokens,
    elapsedSec: el,
    tokensPerSec: tokens / trainSec,
    bestValLoss: best,
    bestStep,
    finalValLossEma: lastValEma,
    finalValLossRaw: lastValRaw,
    numParams: P,
    sustainedGflops: (tokens * trainFlopsPerToken(cfg)) / trainSec / 1e9,
  };
  writeJson(join(runDir, "summary.json"), summary);
  // Final config: the released weights.bin is the best EMA checkpoint on validation loss.
  saveModel(runDir, model, savedConfig({ status: "done", best: bestInfo, summary }), { data: bestW });
  return summary;
}
