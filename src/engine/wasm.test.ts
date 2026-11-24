/**
 * The WASM SIMD GEMM path: the kernel must match the JS kernel, and gradient checks must pass when
 * every product (however small) is routed through WASM. Shapes use K % 4 == 0 so the SIMD path is
 * actually taken (and some with M, N not multiples of the 2x4 tile to exercise the edge tiles).
 */
import { test, after } from "node:test";
import assert from "node:assert/strict";
import {
  Tensor, Rng, gemmNTCore, gemmNT, gemmNTWasm, wasmGemmAvailable, setKernelBackend, matmul, linear,
  causalAttention, noGrad, assembleGemmModule,
} from "./index.js";
import { gradCheck } from "./gradcheck.js";
import { Transformer } from "../model/index.js";

function rand(shape: number[], seed: number, s = 1): Tensor {
  const rng = new Rng(seed);
  const t = Tensor.zeros(shape, true);
  for (let i = 0; i < t.size; i++) t.data[i] = (rng.next() * 2 - 1) * s;
  return t;
}

after(() => setKernelBackend("wasm"));

test("the hand-assembled module validates", () => {
  assert.ok(WebAssembly.validate(assembleGemmModule()));
  assert.ok(wasmGemmAvailable(), "WASM SIMD should be available on Node 22");
});

test("WASM GEMM matches the JS kernel, incl. edge tiles, offsets and accumulate", () => {
  for (const [M, N, K] of [[1, 1, 4], [2, 4, 8], [3, 5, 8], [7, 9, 12], [33, 70, 64], [128, 32, 128]]) {
    const A = rand([M + 2, K], 1).data, B = rand([N + 3, K], 2).data;
    const c1 = rand([M * N + 5], 3).data, c2 = c1.slice();
    // Use non-zero offsets into larger buffers.
    gemmNTCore(M, N, K, A, 2 * K, B, 3 * K, c1, 5, true);
    assert.ok(gemmNTWasm(M, N, K, A, 2 * K, B, 3 * K, c2, 5, true));
    for (let i = 0; i < c1.length; i++) assert.ok(Math.abs(c1[i] - c2[i]) < 1e-4, `${M}x${N}x${K} at ${i}: ${c1[i]} vs ${c2[i]}`);
    const d1 = new Float32Array(M * N), d2 = new Float32Array(M * N).fill(9);
    gemmNTCore(M, N, K, A, 0, B, 0, d1, 0, false);
    gemmNTWasm(M, N, K, A, 0, B, 0, d2, 0, false);
    for (let i = 0; i < d1.length; i++) assert.ok(Math.abs(d1[i] - d2[i]) < 1e-4, "overwrite mode");
  }
});

test("WASM path declines K % 4 != 0 and the dispatcher falls back to JS", () => {
  const A = rand([3, 5], 4).data, B = rand([2, 5], 5).data;
  assert.equal(gemmNTWasm(3, 2, 5, A, 0, B, 0, new Float32Array(6), 0, false), false);
  setKernelBackend("wasm", 0);
  const c = new Float32Array(6), ref = new Float32Array(6);
  gemmNT(3, 2, 5, A, 0, B, 0, c, 0, false);
  gemmNTCore(3, 2, 5, A, 0, B, 0, ref, 0, false);
  assert.deepEqual(Array.from(c), Array.from(ref));
});

test("gradcheck through WASM: matmul, linear, attention", () => {
  setKernelBackend("wasm", 0);
  const r1 = gradCheck(([a, b]) => matmul(a, b), [rand([6, 8], 10), rand([8, 5], 11)]);
  assert.ok(r1.ok, r1.failures.join("\n"));
  const r2 = gradCheck(([x, W, b]) => linear(x, W, b), [rand([6, 8], 12), rand([12, 8], 13), rand([12], 14)]);
  assert.ok(r2.ok, r2.failures.join("\n"));
  const B = 2, T = 8, H = 2, D = 8;
  const r3 = gradCheck(([qkv]) => causalAttention(qkv, B, T, H).out, [rand([B * T, 3 * D], 15, 1.5)], { maxPerInput: 200 });
  assert.ok(r3.ok, r3.failures.join("\n"));
});

test("attention gives the same result on the WASM and JS paths", () => {
  const B = 2, T = 12, H = 2, D = 16;
  const qkv = rand([B * T, 3 * D], 16, 2);
  setKernelBackend("js");
  const a = noGrad(() => causalAttention(qkv, B, T, H));
  setKernelBackend("wasm", 0);
  const b = noGrad(() => causalAttention(qkv, B, T, H));
  for (let i = 0; i < a.out.size; i++) assert.ok(Math.abs(a.out.data[i] - b.out.data[i]) < 1e-5);
  for (let i = 0; i < a.probs.length; i++) assert.ok(Math.abs(a.probs[i] - b.probs[i]) < 1e-5);
});

test("gradcheck through WASM: full transformer", () => {
  setKernelBackend("wasm", 0);
  const cfg = { vocabSize: 13, ctx: 8, dModel: 8, nLayers: 2, nHeads: 2, dMlp: 16 };
  const m = new Transformer(cfg);
  const r = new Rng(21);
  for (const e of m.store.entries) {
    if (e.name.endsWith(".g")) e.tensor.data.fill(1);
    else for (let i = 0; i < e.numel; i++) e.tensor.data[i] = r.normal() * (e.name.endsWith(".b") ? 0.1 : 0.3);
  }
  const B = 2, T = 8;
  const ids = Int32Array.from({ length: B * T }, () => r.int(cfg.vocabSize));
  const targets = Int32Array.from({ length: B * T }, () => r.int(cfg.vocabSize));
  const rep = gradCheck(() => m.forward(ids, B, T, { targets }).loss!, m.store.entries.map((e) => e.tensor), { maxPerInput: 10 });
  assert.ok(rep.ok, rep.failures.slice(0, 8).join("\n"));
});
