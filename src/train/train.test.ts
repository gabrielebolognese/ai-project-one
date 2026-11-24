import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Rng } from "../engine/index.js";
import { loadModel } from "../model/index.js";
import { buildBatch, evalOffsets, sampleOffsets, syntheticTokens } from "./data.js";

import { averageGrads } from "./executor.js";
import { ctxAt, emaDecayAt, lrAt } from "./schedule.js";
import { train, type TrainOptions } from "./trainer.js";

test("LR: warmup, cosine in rho down to the floor", () => {
  const s = { peakLr: 1e-3, minLrFrac: 0.1, warmupSteps: 100, warmupRho: 0.05 };
  assert.ok(lrAt(s, 0, 0) < 2e-5);
  assert.ok(Math.abs(lrAt(s, 49, 0.001) - 0.5e-3) < 2e-5);
  // Warmup also ends by rho, even if few steps were taken (slow steps / short budgets).
  assert.ok(Math.abs(lrAt(s, 3, 0.05) - 1e-3 * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * 0.05)))) < 1e-9);
  assert.ok(Math.abs(lrAt(s, 1000, 0.5) - 1e-3 * 0.55) < 1e-9);
  assert.ok(Math.abs(lrAt(s, 1000, 1) - 1e-4) < 1e-12);
  assert.ok(Math.abs(lrAt(s, 1000, 1.7) - 1e-4) < 1e-12, "rho is clamped");
  let prev = Infinity;
  for (let r = 0.06; r <= 1; r += 0.01) {
    const v = lrAt(s, 1000, r);
    assert.ok(v <= prev + 1e-15, "monotone decay after warmup");
    prev = v;
  }
});

test("context curriculum and EMA warm start", () => {
  const c = { enabled: true, startCtx: 64, fullCtx: 128, switchRho: 0.3 };
  assert.equal(ctxAt(c, 0.1), 64);
  assert.equal(ctxAt(c, 0.3), 128);
  assert.equal(ctxAt({ ...c, enabled: false }, 0.1), 128);
  assert.ok(emaDecayAt(0.999, 0) < 0.2);
  assert.equal(emaDecayAt(0.999, 1e6), 0.999);
});

test("batches are next-token shifted windows; eval windows are fixed and in range", () => {
  const tok = Uint16Array.from({ length: 100 }, (_, i) => i);
  const { ids, targets } = buildBatch(tok, [3, 50], 4);
  assert.deepEqual(Array.from(ids), [3, 4, 5, 6, 50, 51, 52, 53]);
  assert.deepEqual(Array.from(targets), [4, 5, 6, 7, 51, 52, 53, 54]);
  const offs = sampleOffsets(new Rng(1), 100, 8, 500);
  assert.ok(offs.every((o) => o >= 0 && o + 8 < 100));
  const ev = evalOffsets(1000, 16, 320);
  assert.equal(ev.length, 20);
  assert.ok(ev.every((o) => o >= 0 && o + 16 < 1000));
  for (let i = 1; i < ev.length; i++) assert.ok(ev[i] - ev[i - 1] >= 16, "non-overlapping");
  assert.deepEqual(evalOffsets(1000, 16, 320), ev, "deterministic");
});

test("synthetic Markov stream is deterministic and in range", () => {
  const a = syntheticTokens(5000, 50, 9), b = syntheticTokens(5000, 50, 9);
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.ok(a.every((t) => t < 50));
});

test("averageGrads is the mean over replicas", () => {
  const out = new Float32Array(3);
  averageGrads([Float32Array.from([1, 2, 3]), Float32Array.from([3, 2, 1])], out);
  assert.deepEqual(Array.from(out), [2, 2, 2]);
});

function smokeOpts(dir: string, workers: number): TrainOptions {
  return {
    run: "smoke", modelsDir: dir, budgetMin: 5, maxSteps: 60, workers, batchPerWorker: 4,
    lr: { peakLr: 5e-3, minLrFrac: 0.1, warmupSteps: 5, warmupRho: 0.01 },
    ctx: { enabled: false, startCtx: 16, fullCtx: 32, switchRho: 0.3 },
    weightDecay: 0.0, beta1: 0.9, beta2: 0.95, clip: 1, emaDecay: 0.9, evalEveryMin: 100, evalTokens: 2048,
    logEverySec: 0, seed: 3,
  };
}

const smokeCfg = { vocabSize: 32, ctx: 32, dModel: 32, nLayers: 2, nHeads: 2, dMlp: 64 };

/** Easy learnable stream: next = (prev + 7) mod 32, with 10% uniform noise (entropy ~0.6 nats). */
function bigramStream(n: number, seed: number): Uint16Array {
  const r = new Rng(seed);
  const out = new Uint16Array(new SharedArrayBuffer(n * 2));
  let prev = 0;
  for (let i = 0; i < n; i++) {
    prev = r.next() < 0.1 ? r.int(32) : (prev + 7) % 32;
    out[i] = prev;
  }
  return out;
}

test("trainer (in-process): loss drops on a learnable stream; writes log, config, weights, summary", async () => {
  const dir = mkdtempSync(join(tmpdir(), "jal-train-"));
  try {
    const all = bigramStream(60_000, 5);
    const s = await train(smokeCfg, { train: all.subarray(0, 50_000), val: all.subarray(50_000) }, smokeOpts(dir, 0));
    assert.equal(s.steps, 60);
    assert.ok(s.bestValLoss < Math.log(32) - 1.0, `val loss ${s.bestValLoss} should beat uniform ${Math.log(32)}`);
    const run = join(dir, "smoke");
    for (const f of ["config.json", "weights.bin", "last.bin", "log.jsonl", "summary.json"]) assert.ok(existsSync(join(run, f)), f);
    const lines = readFileSync(join(run, "log.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l) as Record<string, number>);
    assert.ok(lines.length >= 60);
    for (const k of ["t", "rho", "step", "tokens", "lr", "loss", "tok_s"]) assert.ok(k in lines[0], `log has ${k}`);
    assert.ok(lines.some((l) => "val_loss" in l));
    const { model, config } = loadModel(run);
    assert.equal(model.numParams, s.numParams);
    assert.equal(config.status, "done");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("trainer (2 worker threads, data-parallel) runs and learns", async () => {
  const dir = mkdtempSync(join(tmpdir(), "jal-train-w-"));
  try {
    const all = bigramStream(60_000, 6);
    const s = await train(smokeCfg, { train: all.subarray(0, 50_000), val: all.subarray(50_000) }, { ...smokeOpts(dir, 2), maxSteps: 40 });
    assert.equal(s.steps, 40);
    assert.equal(s.tokens, 40 * 2 * 4 * 32);
    assert.ok(s.bestValLoss < Math.log(32) - 1.0, `val loss ${s.bestValLoss}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
