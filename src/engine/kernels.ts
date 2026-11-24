/**
 * Hand-written dense kernels over Float32Array (row-major, contiguous).
 *
 * `gemmNT` dispatches to the WASM SIMD kernel (about 9 GFLOPS/thread on the Ryzen 5700U) or to
 * the plain-JS kernel `gemmNTCore` (about 1.4 GFLOPS/thread); see results/bench.json.
 *
 * The only GEMM core is `gemmNTCore`: C[i,j] (+)= sum_k A[i,k] * B[j,k], i.e. both operands are
 * read along contiguous rows (dot-product form). It uses a 4x4 register tile: 8 loads feed 16
 * multiply-adds per k, which is the best ratio we found for V8 (it does not auto-vectorise, so
 * the win comes from register reuse and fewer bounds checks). Every other layout is reduced to
 * NT by a cache-blocked transpose, which costs O(MK + NK) against O(MNK) multiply-adds.
 * See scripts/bench.ts for the measured comparison against a naive i-k-j loop.
 */

import { gemmNTWasm, wasmGemmAvailable } from "./wasm-gemm.js";

export type KernelBackend = "wasm" | "js";

let backend: KernelBackend = process.env.JAL_KERNEL === "js" || !wasmGemmAvailable() ? "js" : "wasm";
/** Products with fewer multiply-adds than this use the JS kernel (WASM call + copy overhead). */
let wasmMinWork = 4096;

/** Select the GEMM backend (benchmarks, tests). `minWork` = M*N*K below which JS is used. */
export function setKernelBackend(b: KernelBackend, minWork = 4096): void {
  if (b === "wasm" && !wasmGemmAvailable()) throw new Error("WASM SIMD is not available in this runtime");
  backend = b;
  wasmMinWork = minWork;
}

export function kernelBackend(): KernelBackend {
  return backend;
}

/**
 * Dispatching NT GEMM: C[M,N] (+)= A[M,K] * B[N,K]^T. Uses the hand-assembled WASM SIMD kernel
 * (wasm-gemm.ts) when selected and K % 4 == 0, else the JS register-tiled kernel below.
 */
export function gemmNT(
  M: number, N: number, K: number,
  A: Float32Array, aOff: number,
  B: Float32Array, bOff: number,
  C: Float32Array, cOff: number,
  accumulate: boolean,
): void {
  if (backend === "wasm" && M * N * K >= wasmMinWork && gemmNTWasm(M, N, K, A, aOff, B, bOff, C, cOff, accumulate)) return;
  gemmNTCore(M, N, K, A, aOff, B, bOff, C, cOff, accumulate);
}

/** Out-of-place transpose: src is [rows, cols], dst becomes [cols, rows]. Blocked for cache reuse. */
export function transpose(src: Float32Array, rows: number, cols: number, dst?: Float32Array): Float32Array {
  const out = dst ?? new Float32Array(rows * cols);
  const BS = 32;
  for (let i0 = 0; i0 < rows; i0 += BS) {
    const i1 = Math.min(i0 + BS, rows);
    for (let j0 = 0; j0 < cols; j0 += BS) {
      const j1 = Math.min(j0 + BS, cols);
      for (let i = i0; i < i1; i++) {
        const ro = i * cols;
        for (let j = j0; j < j1; j++) out[j * rows + i] = src[ro + j];
      }
    }
  }
  return out;
}

/**
 * C[M,N] (+)= A[M,K] * B[N,K]^T. If `accumulate` is false C is overwritten.
 * Offsets allow operating on sub-matrices of larger buffers (rows must still be contiguous).
 */
export function gemmNTCore(
  M: number, N: number, K: number,
  A: Float32Array, aOff: number,
  B: Float32Array, bOff: number,
  C: Float32Array, cOff: number,
  accumulate: boolean,
): void {
  const M4 = M - (M % 4);
  const N4 = N - (N % 4);
  for (let i = 0; i < M4; i += 4) {
    const a0 = aOff + i * K, a1 = a0 + K, a2 = a1 + K, a3 = a2 + K;
    const r0 = cOff + i * N, r1 = r0 + N, r2 = r1 + N, r3 = r2 + N;
    for (let j = 0; j < N4; j += 4) {
      const b0 = bOff + j * K, b1 = b0 + K, b2 = b1 + K, b3 = b2 + K;
      let c00 = 0, c01 = 0, c02 = 0, c03 = 0;
      let c10 = 0, c11 = 0, c12 = 0, c13 = 0;
      let c20 = 0, c21 = 0, c22 = 0, c23 = 0;
      let c30 = 0, c31 = 0, c32 = 0, c33 = 0;
      for (let k = 0; k < K; k++) {
        const x0 = A[a0 + k], x1 = A[a1 + k], x2 = A[a2 + k], x3 = A[a3 + k];
        const y0 = B[b0 + k], y1 = B[b1 + k], y2 = B[b2 + k], y3 = B[b3 + k];
        c00 += x0 * y0; c01 += x0 * y1; c02 += x0 * y2; c03 += x0 * y3;
        c10 += x1 * y0; c11 += x1 * y1; c12 += x1 * y2; c13 += x1 * y3;
        c20 += x2 * y0; c21 += x2 * y1; c22 += x2 * y2; c23 += x2 * y3;
        c30 += x3 * y0; c31 += x3 * y1; c32 += x3 * y2; c33 += x3 * y3;
      }
      if (accumulate) {
        C[r0 + j] += c00; C[r0 + j + 1] += c01; C[r0 + j + 2] += c02; C[r0 + j + 3] += c03;
        C[r1 + j] += c10; C[r1 + j + 1] += c11; C[r1 + j + 2] += c12; C[r1 + j + 3] += c13;
        C[r2 + j] += c20; C[r2 + j + 1] += c21; C[r2 + j + 2] += c22; C[r2 + j + 3] += c23;
        C[r3 + j] += c30; C[r3 + j + 1] += c31; C[r3 + j + 2] += c32; C[r3 + j + 3] += c33;
      } else {
        C[r0 + j] = c00; C[r0 + j + 1] = c01; C[r0 + j + 2] = c02; C[r0 + j + 3] = c03;
        C[r1 + j] = c10; C[r1 + j + 1] = c11; C[r1 + j + 2] = c12; C[r1 + j + 3] = c13;
        C[r2 + j] = c20; C[r2 + j + 1] = c21; C[r2 + j + 2] = c22; C[r2 + j + 3] = c23;
        C[r3 + j] = c30; C[r3 + j + 1] = c31; C[r3 + j + 2] = c32; C[r3 + j + 3] = c33;
      }
    }
    // Remaining columns for these 4 rows.
    for (let j = N4; j < N; j++) {
      const b0 = bOff + j * K;
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let k = 0; k < K; k++) {
        const y = B[b0 + k];
        s0 += A[a0 + k] * y; s1 += A[a1 + k] * y; s2 += A[a2 + k] * y; s3 += A[a3 + k] * y;
      }
      if (accumulate) { C[r0 + j] += s0; C[r1 + j] += s1; C[r2 + j] += s2; C[r3 + j] += s3; }
      else { C[r0 + j] = s0; C[r1 + j] = s1; C[r2 + j] = s2; C[r3 + j] = s3; }
    }
  }
  // Remaining rows.
  for (let i = M4; i < M; i++) {
    const a0 = aOff + i * K;
    const r0 = cOff + i * N;
    for (let j = 0; j < N; j++) {
      const b0 = bOff + j * K;
      let s = 0;
      for (let k = 0; k < K; k++) s += A[a0 + k] * B[b0 + k];
      if (accumulate) C[r0 + j] += s; else C[r0 + j] = s;
    }
  }
}

/**
 * General GEMM: C[M,N] (+)= op(A) * op(B).
 *  - transA=false: A is stored [M,K]; transA=true: A is stored [K,M].
 *  - transB=false: B is stored [K,N]; transB=true: B is stored [N,K].
 */
export function gemm(
  transA: boolean, transB: boolean,
  M: number, N: number, K: number,
  A: Float32Array, B: Float32Array, C: Float32Array,
  accumulate = false,
): void {
  const a = transA ? transpose(A, K, M) : A;
  const b = transB ? B : transpose(B, K, N);
  gemmNT(M, N, K, a, 0, b, 0, C, 0, accumulate);
}

/** Naive i-k-j reference GEMM (C = A[M,K] * B[K,N]); used by tests and the benchmark baseline. */
export function gemmNaive(M: number, N: number, K: number, A: Float32Array, B: Float32Array, C: Float32Array): void {
  C.fill(0);
  for (let i = 0; i < M; i++) {
    const co = i * N;
    for (let k = 0; k < K; k++) {
      const a = A[i * K + k];
      const bo = k * N;
      for (let j = 0; j < N; j++) C[co + j] += a * B[bo + j];
    }
  }
}
