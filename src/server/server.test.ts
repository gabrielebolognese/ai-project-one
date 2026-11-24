import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, copyFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { noGrad } from "../engine/index.js";
import { Transformer, makeConfig, saveModel } from "../model/index.js";
import { IncrementalDecoder } from "./decoder.js";
import { fitPca, project } from "./pca.js";
import { downsample } from "./http.js";
import { ModelHost } from "./model-host.js";
import { generate } from "./inference.js";

function smallModel(seed = 7): Transformer {
  const m = new Transformer(makeConfig("small", 300, { ctx: 24 }));
  m.init(seed);
  // make LN gains/biases non-trivial so the check covers them
  for (const e of m.store.entries) if (e.name.endsWith(".b") || e.name.endsWith(".g")) {
    for (let i = 0; i < e.numel; i++) e.tensor.data[i] += 0.05 * Math.sin(i + e.offset);
  }
  return m;
}

test("KV-cached decoder matches Transformer.forward logits", () => {
  const m = smallModel();
  const V = m.cfg.vocabSize;
  const ids = Array.from({ length: 24 }, (_, i) => (i * 37 + 11) % V);
  const dec = new IncrementalDecoder(m);
  let got = dec.prefill(ids.slice(0, 5));
  for (let n = 5; n <= 24; n++) {
    const ref = noGrad(() => m.forward(ids.slice(0, n), 1, n).logits.data).slice((n - 1) * V, n * V);
    let maxDiff = 0;
    for (let i = 0; i < V; i++) maxDiff = Math.max(maxDiff, Math.abs(ref[i] - got[i]));
    assert.ok(maxDiff < 1e-4, `n=${n}: max |diff| ${maxDiff}`);
    if (n < 24) got = dec.step(ids[n]);
  }
  assert.ok(dec.full);
  assert.throws(() => dec.step(1));
});

test("PCA recovers the dominant direction", () => {
  const d = 6;
  const rows: Float32Array[] = [];
  for (let i = 0; i < 200; i++) {
    const a = Math.sin(i * 1.7) * 5, b = Math.cos(i * 0.9) * 1;
    const r = new Float32Array(d);
    r[2] = a; r[4] = b; r[0] = 0.01 * Math.sin(i);
    rows.push(r);
  }
  const p = fitPca(rows, d, 2);
  assert.ok(Math.abs(p.components[2]) > 0.99, "first PC ~ axis 2");
  assert.ok(Math.abs(p.components[d + 4]) > 0.99, "second PC ~ axis 4");
  assert.ok(p.explained[0] > p.explained[1]);
  const flat = new Float32Array(d * 2);
  flat.set(rows[0], 0); flat.set(rows[1], d);
  assert.equal(project(p, flat, 2).length, 4);
});

test("downsample block-pools to the requested size", () => {
  const data = Float32Array.from({ length: 10 * 6 }, (_, i) => i);
  const ds = downsample(data, 10, 6, 5, 3, "mean");
  assert.equal(ds.rows, 5);
  assert.equal(ds.cols, 3);
  assert.equal(ds.data[0], (0 + 1 + 6 + 7) / 4);
  const am = downsample(Float32Array.from([1, -3, 2, 0]), 2, 2, 1, 1, "absmax");
  assert.equal(am.data[0], -3);
});

test("ModelHost loads the newest run, prefers release, falls back to random init", async () => {
  const root = mkdtempSync(join(tmpdir(), "jal-host-"));
  try {
    mkdirSync(join(root, "models"), { recursive: true });
    copyFileSync(join(process.cwd(), "models", "tokenizer.json"), join(root, "models", "tokenizer.json"));
    const quiet = { root, log: () => {} };
    const r0 = new ModelHost(quiet);
    assert.equal(r0.source.kind, "random");
    assert.ok(r0.warnings.some((w) => w.includes("RANDOMLY")));

    const V = r0.tokenizer.vocabSize;
    const m = new Transformer(makeConfig("small", V, { ctx: 32 }));
    m.init(3);
    saveModel(join(root, "models", "runA"), m);
    const r1 = new ModelHost(quiet);
    assert.equal(r1.source.kind, "run");
    assert.equal(r1.model.cfg.dModel, 96);
    assert.equal(r1.model.store.data[5], m.store.data[5]);

    // generation crosses the context limit (prompt 30 + 20 tokens > ctx 32) via context shift
    const g = await generate(r1, Array.from({ length: 30 }, (_, i) => 3 + i), { maxTokens: 20, stopAtNewline: false });
    assert.ok(g.steps.length >= 1 && g.steps.length <= 20);

    saveModel(join(root, "models", "release"), m);
    assert.equal(new ModelHost(quiet).source.kind, "release");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
