import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AdamW, Rng, backward, clipGradNorm, noGrad } from "../engine/index.js";
import { gradCheck } from "../engine/gradcheck.js";
import { Transformer, countParams, loadModel, makeConfig, saveModel, type ModelConfig } from "./index.js";

const tiny: ModelConfig = { vocabSize: 11, ctx: 8, dModel: 8, nLayers: 2, nHeads: 2, dMlp: 16 };

function randomIds(n: number, V: number, seed: number): Int32Array {
  const r = new Rng(seed);
  return Int32Array.from({ length: n }, () => r.int(V));
}

test("parameter count matches the formula and the SPEC (~1.07M for the base preset)", () => {
  const m = new Transformer(tiny);
  assert.equal(m.numParams, countParams(tiny));
  const base = makeConfig("base", 2048);
  const n = countParams(base);
  assert.ok(n > 1.0e6 && n < 1.1e6, `base preset has ${n} params`);
});

test("gradcheck: full transformer loss w.r.t. every parameter tensor", () => {
  const m = new Transformer(tiny);
  m.init(3);
  // Larger-than-default init so the check is not dominated by near-zero gradients.
  const r = new Rng(5);
  for (const e of m.store.entries) {
    if (!e.name.endsWith(".g") && !e.name.endsWith(".b")) for (let i = 0; i < e.numel; i++) e.tensor.data[i] = r.normal() * 0.3;
    else if (e.name.endsWith(".b")) for (let i = 0; i < e.numel; i++) e.tensor.data[i] = r.normal() * 0.1;
  }
  const B = 2, T = 5;
  const ids = randomIds(B * T, tiny.vocabSize, 9);
  const targets = randomIds(B * T, tiny.vocabSize, 10);
  const params = m.store.entries.map((e) => e.tensor);
  const report = gradCheck(() => m.forward(ids, B, T, { targets }).loss!, params, { maxPerInput: 12, eps: 1e-2 });
  assert.ok(report.ok, report.failures.slice(0, 8).join("\n"));
  assert.ok(report.checked >= params.length * 3);
});

test("causal: logits at position i do not depend on tokens after i", () => {
  const m = new Transformer(tiny);
  m.init(4);
  const T = 8;
  const ids = randomIds(T, tiny.vocabSize, 11);
  const a = noGrad(() => m.forward(ids, 1, T).logits.data.slice());
  const ids2 = ids.slice();
  ids2[6] = (ids2[6] + 1) % tiny.vocabSize;
  ids2[7] = (ids2[7] + 3) % tiny.vocabSize;
  const b = noGrad(() => m.forward(ids2, 1, T).logits.data);
  const V = tiny.vocabSize;
  for (let i = 0; i < 6 * V; i++) assert.equal(b[i], a[i], `position ${Math.floor(i / V)} changed`);
});

test("trace exposes residuals, attention, MLP activations and logits with the right shapes", () => {
  const m = new Transformer(tiny);
  m.init(5);
  const B = 1, T = 6;
  const { trace, logits } = noGrad(() => m.forward(randomIds(T, tiny.vocabSize, 12), B, T, { trace: true }));
  assert.ok(trace);
  assert.equal(trace.layers.length, tiny.nLayers);
  for (const L of trace.layers) {
    assert.equal(L.residIn.length, T * tiny.dModel);
    assert.equal(L.residMid.length, T * tiny.dModel);
    assert.equal(L.residOut.length, T * tiny.dModel);
    assert.equal(L.attnProbs.length, tiny.nHeads * T * T);
    assert.equal(L.mlpAct.length, T * tiny.dMlp);
  }
  // Residual bookkeeping: out = mid + mlpOut, mid = in + attnOut.
  const L0 = trace.layers[0];
  for (let i = 0; i < T * tiny.dModel; i++) {
    assert.ok(Math.abs(L0.residMid[i] - (L0.residIn[i] + L0.attnOut[i])) < 1e-6);
    assert.ok(Math.abs(L0.residOut[i] - (L0.residMid[i] + L0.mlpOut[i])) < 1e-6);
  }
  assert.equal(trace.logits, logits.data);
  assert.equal(trace.layers[1].residIn, trace.layers[0].residOut);
});

test("overfit: loss on a single fixed batch goes near zero", () => {
  const cfg: ModelConfig = { vocabSize: 16, ctx: 16, dModel: 32, nLayers: 2, nHeads: 2, dMlp: 64 };
  const m = new Transformer(cfg);
  m.init(6);
  const B = 2, T = 16;
  const seq = randomIds(B * (T + 1), cfg.vocabSize, 13);
  const ids = new Int32Array(B * T), targets = new Int32Array(B * T);
  for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
    ids[b * T + t] = seq[b * (T + 1) + t];
    targets[b * T + t] = seq[b * (T + 1) + t + 1];
  }
  const opt = new AdamW(m.store.size, m.store.decayMask(), { weightDecay: 0 });
  let first = 0, last = 0;
  for (let step = 0; step < 800; step++) {
    m.store.zeroGrad();
    const { loss } = m.forward(ids, B, T, { targets });
    backward(loss!);
    clipGradNorm(m.store.grad, 1);
    opt.update(m.store.data, m.store.grad, 1e-2);
    if (step === 0) first = loss!.data[0];
    last = loss!.data[0];
    if (last < 0.02) break;
  }
  assert.ok(first > 2, `initial loss ${first} should be ~ln(16)=2.77`);
  assert.ok(last < 0.05, `loss after overfitting is ${last}`);
});

test("saveModel / loadModel round-trip gives identical logits", () => {
  const m = new Transformer(tiny);
  m.init(7);
  const dir = mkdtempSync(join(tmpdir(), "jal-model-"));
  try {
    saveModel(dir, m, { note: "test" });
    const { model, config } = loadModel(dir);
    assert.equal(config.note, "test");
    assert.equal(config.tensors.length, m.store.entries.length);
    const ids = randomIds(6, tiny.vocabSize, 14);
    const a = noGrad(() => m.forward(ids, 1, 6).logits.data);
    const b = noGrad(() => model.forward(ids, 1, 6).logits.data);
    assert.deepEqual(Array.from(b), Array.from(a));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
