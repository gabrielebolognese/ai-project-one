import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  Tensor, Rng, add, mul, scale, matmul, linear, gelu, softmax, layernorm, embedding, causalAttention,
  crossEntropy, sum, gemm, gemmNaive, transpose, AdamW, clipGradNorm, gradNorm, ParamStore,
  saveCheckpoint, loadCheckpoint, loadIntoStore, backward, noGrad, tapeLength, clearTape,
} from "./index.js";
import { gradCheck, type GradCheckReport } from "./gradcheck.js";

function rand(shape: number[], seed: number, s = 1, name = ""): Tensor {
  const rng = new Rng(seed);
  const t = Tensor.zeros(shape, true, name);
  for (let i = 0; i < t.size; i++) t.data[i] = (rng.next() * 2 - 1) * s;
  return t;
}

function expectOk(r: GradCheckReport, label: string): void {
  assert.ok(r.checked > 0, `${label}: nothing checked`);
  assert.ok(r.ok, `${label}: ${r.failures.length} gradient mismatches, e.g.\n${r.failures.slice(0, 5).join("\n")}`);
}

// ------------------------------------------------------------------ kernels

test("gemm matches naive reference for all transpose modes and odd shapes", () => {
  for (const [M, N, K] of [[1, 1, 1], [3, 5, 7], [4, 4, 4], [9, 6, 13], [17, 33, 8]]) {
    const A = rand([M, K], 1).data, B = rand([K, N], 2).data;
    const ref = new Float32Array(M * N);
    gemmNaive(M, N, K, A, B, ref);
    const AT = transpose(A, M, K), BT = transpose(B, K, N);
    const cases: [boolean, boolean, Float32Array, Float32Array][] = [
      [false, false, A, B], [true, false, AT, B], [false, true, A, BT], [true, true, AT, BT],
    ];
    for (const [ta, tb, a, b] of cases) {
      const C = new Float32Array(M * N);
      gemm(ta, tb, M, N, K, a, b, C);
      for (let i = 0; i < M * N; i++) assert.ok(Math.abs(C[i] - ref[i]) < 1e-4, `gemm ${ta}/${tb} ${M}x${N}x${K} at ${i}`);
      // Accumulate mode adds on top.
      gemm(ta, tb, M, N, K, a, b, C, true);
      for (let i = 0; i < M * N; i++) assert.ok(Math.abs(C[i] - 2 * ref[i]) < 2e-4, "gemm accumulate");
    }
  }
});

test("transpose round-trips", () => {
  const a = rand([37, 70], 3).data;
  const b = transpose(transpose(a, 37, 70), 70, 37);
  assert.deepEqual(Array.from(b), Array.from(a));
});

// ------------------------------------------------------------------ gradient checks, one per op

test("gradcheck: matmul (non-multiple-of-4 shapes)", () => {
  expectOk(gradCheck(([a, b]) => matmul(a, b), [rand([5, 7], 10), rand([7, 6], 11)]), "matmul");
});

test("gradcheck: linear with bias, 3-D input", () => {
  expectOk(gradCheck(([x, W, b]) => linear(x, W, b), [rand([2, 3, 5], 12), rand([6, 5], 13), rand([6], 14)]), "linear");
});

test("gradcheck: linear without bias (tied head form)", () => {
  expectOk(gradCheck(([x, W]) => linear(x, W, null), [rand([5, 4], 15), rand([9, 4], 16)]), "linear-nobias");
});

test("gradcheck: add (same shape and broadcast bias)", () => {
  expectOk(gradCheck(([a, b]) => add(a, b), [rand([3, 4], 17), rand([3, 4], 18)]), "add");
  expectOk(gradCheck(([a, b]) => add(a, b), [rand([3, 4], 19), rand([4], 20)]), "add-broadcast");
});

test("gradcheck: mul and scale", () => {
  expectOk(gradCheck(([a, b]) => mul(a, b), [rand([4, 5], 21), rand([4, 5], 22)]), "mul");
  expectOk(gradCheck(([a]) => scale(a, -1.7), [rand([3, 3], 23)]), "scale");
});

test("gradcheck: gelu", () => {
  expectOk(gradCheck(([a]) => gelu(a), [rand([4, 8], 24, 3)]), "gelu");
});

test("gradcheck: softmax", () => {
  expectOk(gradCheck(([a]) => softmax(a), [rand([3, 7], 25, 2)]), "softmax");
});

test("gradcheck: sum", () => {
  expectOk(gradCheck(([a]) => sum(a), [rand([3, 4], 26)]), "sum");
});

test("gradcheck: layernorm (x, gain, bias)", () => {
  const g = rand([6], 28);
  for (let i = 0; i < 6; i++) g.data[i] += 1.5;
  expectOk(gradCheck(([x, gg, b]) => layernorm(x, gg, b), [rand([4, 6], 27, 2), g, rand([6], 29)]), "layernorm");
});

test("gradcheck: embedding with repeated ids", () => {
  const ids = Int32Array.from([3, 1, 3, 0, 4, 3]);
  expectOk(gradCheck(([w]) => embedding(ids, w), [rand([5, 4], 30)]), "embedding");
});

test("gradcheck: fused causal attention", () => {
  const B = 2, T = 5, H = 2, D = 6;
  expectOk(gradCheck(([qkv]) => causalAttention(qkv, B, T, H).out, [rand([B * T, 3 * D], 31, 1.5)]), "attention");
});

test("gradcheck: cross-entropy (with an ignored target)", () => {
  const targets = Int32Array.from([2, 0, -1, 4]);
  expectOk(gradCheck(([z]) => crossEntropy(z, targets), [rand([4, 5], 32, 2)]), "cross-entropy");
});

test("cross-entropy value matches a hand computation", () => {
  const z = Tensor.from([1, 2, 3, 0, 0, 0], [2, 3]);
  const l = crossEntropy(z, Int32Array.from([2, 1])).data[0];
  const l0 = -Math.log(Math.exp(3) / (Math.exp(1) + Math.exp(2) + Math.exp(3)));
  const l1 = -Math.log(1 / 3);
  assert.ok(Math.abs(l - (l0 + l1) / 2) < 1e-5);
});

// ------------------------------------------------------------------ attention mask

test("attention probabilities are causal and rows sum to 1", () => {
  const B = 2, T = 7, H = 3, D = 9;
  const { probs } = noGrad(() => causalAttention(rand([B * T, 3 * D], 40, 2), B, T, H));
  for (let bh = 0; bh < B * H; bh++) {
    for (let i = 0; i < T; i++) {
      let s = 0;
      for (let j = 0; j < T; j++) {
        const p = probs[bh * T * T + i * T + j];
        if (j > i) assert.equal(p, 0, `future key ${j} visible from query ${i}`);
        s += p;
      }
      assert.ok(Math.abs(s - 1) < 1e-5, "row does not sum to 1");
    }
  }
});

test("attention output at position i ignores inputs at positions > i", () => {
  const B = 1, T = 6, H = 2, D = 4;
  const qkv = rand([B * T, 3 * D], 41);
  const before = noGrad(() => causalAttention(qkv, B, T, H).out.data.slice());
  // Perturb every q/k/v entry of the last two positions.
  for (let i = 4 * 3 * D; i < T * 3 * D; i++) qkv.data[i] += 5;
  const after = noGrad(() => causalAttention(qkv, B, T, H).out.data);
  for (let i = 0; i < 4 * D; i++) assert.equal(after[i], before[i]);
  let changed = false;
  for (let i = 4 * D; i < T * D; i++) if (after[i] !== before[i]) changed = true;
  assert.ok(changed, "perturbation should change later positions");
});

// ------------------------------------------------------------------ tape / no-grad

test("noGrad records nothing; backward clears the tape", () => {
  clearTape();
  const a = rand([2, 2], 50);
  noGrad(() => gelu(a));
  assert.equal(tapeLength(), 0);
  const l = sum(gelu(a));
  assert.ok(tapeLength() > 0);
  backward(l);
  assert.equal(tapeLength(), 0);
});

// ------------------------------------------------------------------ optimiser

test("AdamW first step matches the closed form (decay on masked elements only)", () => {
  const p = Float32Array.from([1, -2]);
  const g = Float32Array.from([0.5, -0.25]);
  const opt = new AdamW(2, Uint8Array.from([1, 0]), { beta1: 0.9, beta2: 0.95, eps: 1e-8, weightDecay: 0.1 });
  opt.update(p, g, 0.01);
  // Step 1: m_hat = g, v_hat = g^2 -> update = lr * sign(g) (up to eps).
  assert.ok(Math.abs(p[0] - (1 * (1 - 0.01 * 0.1) - 0.01)) < 1e-6);
  assert.ok(Math.abs(p[1] - (-2 + 0.01)) < 1e-6);
});

test("clipGradNorm caps the global norm", () => {
  const g = Float32Array.from([3, 4]);
  const n = clipGradNorm(g, 1);
  assert.ok(Math.abs(n - 5) < 1e-6);
  assert.ok(Math.abs(gradNorm(g) - 1) < 1e-6);
  const h = Float32Array.from([0.3, 0.4]);
  clipGradNorm(h, 1);
  assert.ok(Math.abs(h[0] - 0.3) < 1e-7);
});

// ------------------------------------------------------------------ params + checkpoint I/O

test("ParamStore views share one flat buffer; checkpoint round-trips with a tensor table", () => {
  const specs = [
    { name: "a", shape: [2, 3], decay: true },
    { name: "b", shape: [3], decay: false },
  ];
  const s = new ParamStore(specs);
  assert.equal(s.size, 9);
  s.get("b").data[1] = 7;
  assert.equal(s.data[7], 7);
  s.get("a").g()[0] = 2;
  assert.equal(s.grad[0], 2);
  assert.deepEqual(Array.from(s.decayMask()), [1, 1, 1, 1, 1, 1, 0, 0, 0]);
  for (let i = 0; i < s.size; i++) s.data[i] = i * 0.5;

  const dir = mkdtempSync(join(tmpdir(), "jal-ckpt-"));
  try {
    saveCheckpoint(dir, s, { hello: "world" });
    const { config, data } = loadCheckpoint<{ hello: string }>(dir);
    assert.equal(config.hello, "world");
    assert.deepEqual(config.tensors.map((t) => [t.name, t.shape, t.offset, t.numel]), [["a", [2, 3], 0, 6], ["b", [3], 6, 3]]);
    const s2 = new ParamStore(specs);
    loadIntoStore(s2, config.tensors, data);
    assert.deepEqual(Array.from(s2.data), Array.from(s.data));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
