/**
 * Differentiable ops. Each forward computes into a fresh Float32Array and, when needed, records a
 * backward closure that accumulates into the inputs' gradients. Heavy ops (LayerNorm, attention,
 * cross-entropy) are fused: one forward kernel, one backward kernel, no per-element graph nodes.
 */
import { Tensor, needsGrad, record } from "./tensor.js";
import { gemmNT, transpose } from "./kernels.js";

export type IntArray = Int32Array | Uint16Array | Uint32Array | number[];

function assert(cond: boolean, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}

function rows(t: Tensor): number {
  return t.size / t.shape[t.shape.length - 1];
}

function lastDim(t: Tensor): number {
  return t.shape[t.shape.length - 1];
}

// ---------------------------------------------------------------------------------------------
// Matrix products
// ---------------------------------------------------------------------------------------------

/** C[M,N] = A[M,K] @ B[K,N]. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  assert(a.shape.length === 2 && b.shape.length === 2, "matmul expects 2-D tensors");
  const [M, K] = a.shape;
  const [K2, N] = b.shape;
  assert(K === K2, `matmul inner dims differ: ${K} vs ${K2}`);
  const bT = transpose(b.data, K, N); // [N,K]
  const out = Tensor.zeros([M, N]);
  gemmNT(M, N, K, a.data, 0, bT, 0, out.data, 0, false);
  if (needsGrad(a, b)) {
    out.requiresGrad = true;
    record(() => {
      const dC = out.grad;
      if (!dC) return;
      if (a.requiresGrad) {
        // dA[M,K] += dC[M,N] @ B^T  -> NT with B stored [K,N] as the "rows" operand.
        gemmNT(M, K, N, dC, 0, b.data, 0, a.g(), 0, true);
      }
      if (b.requiresGrad) {
        // dB[K,N] += A^T @ dC -> NT core with A^T [K,M] and dC^T [N,M].
        const aT = transpose(a.data, M, K);
        const dCT = transpose(dC, M, N);
        gemmNT(K, N, M, aT, 0, dCT, 0, b.g(), 0, true);
      }
    });
  }
  return out;
}

/**
 * Affine layer y[M,out] = x[M,in] @ W[out,in]^T + b[out] (PyTorch weight layout).
 * x may have any leading shape; the last dim must equal `in`.
 */
export function linear(x: Tensor, W: Tensor, b: Tensor | null): Tensor {
  const [O, I] = W.shape;
  assert(lastDim(x) === I, `linear: input dim ${lastDim(x)} != weight in-dim ${I}`);
  const M = rows(x);
  const outShape = [...x.shape.slice(0, -1), O];
  const out = Tensor.zeros(outShape);
  gemmNT(M, O, I, x.data, 0, W.data, 0, out.data, 0, false);
  if (b) {
    const y = out.data;
    const bd = b.data;
    for (let i = 0; i < M; i++) {
      const o = i * O;
      for (let j = 0; j < O; j++) y[o + j] += bd[j];
    }
  }
  if (needsGrad(x, W, b)) {
    out.requiresGrad = true;
    record(() => {
      const dY = out.grad;
      if (!dY) return;
      if (x.requiresGrad) {
        // dX[M,I] += dY[M,O] @ W[O,I]  -> NT with W^T [I,O].
        const WT = transpose(W.data, O, I);
        gemmNT(M, I, O, dY, 0, WT, 0, x.g(), 0, true);
      }
      if (W.requiresGrad) {
        // dW[O,I] += dY^T @ X -> NT with dY^T [O,M] and X^T [I,M].
        const dYT = transpose(dY, M, O);
        const xT = transpose(x.data, M, I);
        gemmNT(O, I, M, dYT, 0, xT, 0, W.g(), 0, true);
      }
      if (b && b.requiresGrad) {
        const db = b.g();
        for (let i = 0; i < M; i++) {
          const o = i * O;
          for (let j = 0; j < O; j++) db[j] += dY[o + j];
        }
      }
    });
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------------------------

/**
 * a + b. b must either have the same size as a, or be a vector broadcast over a's last dim.
 */
export function add(a: Tensor, b: Tensor): Tensor {
  const n = a.size;
  const out = Tensor.zeros(a.shape);
  const y = out.data, ad = a.data, bd = b.data;
  const broadcast = b.size !== n;
  if (broadcast) {
    const D = lastDim(a);
    assert(b.size === D, `add: cannot broadcast size ${b.size} over [${a.shape.join(",")}]`);
    for (let i = 0; i < n; i += D) for (let j = 0; j < D; j++) y[i + j] = ad[i + j] + bd[j];
  } else {
    for (let i = 0; i < n; i++) y[i] = ad[i] + bd[i];
  }
  if (needsGrad(a, b)) {
    out.requiresGrad = true;
    record(() => {
      const dy = out.grad;
      if (!dy) return;
      if (a.requiresGrad) {
        const da = a.g();
        for (let i = 0; i < n; i++) da[i] += dy[i];
      }
      if (b.requiresGrad) {
        const db = b.g();
        if (broadcast) {
          const D = db.length;
          for (let i = 0; i < n; i += D) for (let j = 0; j < D; j++) db[j] += dy[i + j];
        } else {
          for (let i = 0; i < n; i++) db[i] += dy[i];
        }
      }
    });
  }
  return out;
}

/** Elementwise a * b (same shape). */
export function mul(a: Tensor, b: Tensor): Tensor {
  const n = a.size;
  assert(b.size === n, "mul: shapes must match");
  const out = Tensor.zeros(a.shape);
  const y = out.data, ad = a.data, bd = b.data;
  for (let i = 0; i < n; i++) y[i] = ad[i] * bd[i];
  if (needsGrad(a, b)) {
    out.requiresGrad = true;
    record(() => {
      const dy = out.grad;
      if (!dy) return;
      if (a.requiresGrad) {
        const da = a.g();
        for (let i = 0; i < n; i++) da[i] += dy[i] * bd[i];
      }
      if (b.requiresGrad) {
        const db = b.g();
        for (let i = 0; i < n; i++) db[i] += dy[i] * ad[i];
      }
    });
  }
  return out;
}

/** Multiply by a constant. */
export function scale(a: Tensor, s: number): Tensor {
  const n = a.size;
  const out = Tensor.zeros(a.shape);
  for (let i = 0; i < n; i++) out.data[i] = a.data[i] * s;
  if (needsGrad(a)) {
    out.requiresGrad = true;
    record(() => {
      const dy = out.grad;
      if (!dy) return;
      const da = a.g();
      for (let i = 0; i < n; i++) da[i] += dy[i] * s;
    });
  }
  return out;
}

const GELU_C = Math.sqrt(2 / Math.PI);

/** GELU, tanh approximation (as in GPT-2). tanh(u) is computed as 1 - 2/(1+e^{2u}) (faster in V8). */
export function gelu(x: Tensor): Tensor {
  const n = x.size;
  const out = Tensor.zeros(x.shape);
  const xd = x.data, y = out.data;
  const rec = needsGrad(x);
  const tanhU = rec ? new Float32Array(n) : null;
  for (let i = 0; i < n; i++) {
    const v = xd[i];
    const t = 1 - 2 / (1 + Math.exp(2 * GELU_C * (v + 0.044715 * v * v * v)));
    if (tanhU) tanhU[i] = t;
    y[i] = 0.5 * v * (1 + t);
  }
  if (rec && tanhU) {
    out.requiresGrad = true;
    record(() => {
      const dy = out.grad;
      if (!dy) return;
      const dx = x.g();
      for (let i = 0; i < n; i++) {
        const v = xd[i];
        const t = tanhU[i];
        const du = GELU_C * (1 + 3 * 0.044715 * v * v);
        dx[i] += dy[i] * (0.5 * (1 + t) + 0.5 * v * (1 - t * t) * du);
      }
    });
  }
  return out;
}

/** Row-wise softmax over the last dimension. */
export function softmax(x: Tensor): Tensor {
  const D = lastDim(x);
  const R = rows(x);
  const out = Tensor.zeros(x.shape);
  const xd = x.data, y = out.data;
  for (let r = 0; r < R; r++) {
    const o = r * D;
    let m = -Infinity;
    for (let j = 0; j < D; j++) if (xd[o + j] > m) m = xd[o + j];
    let s = 0;
    for (let j = 0; j < D; j++) {
      const e = Math.exp(xd[o + j] - m);
      y[o + j] = e;
      s += e;
    }
    const inv = 1 / s;
    for (let j = 0; j < D; j++) y[o + j] *= inv;
  }
  if (needsGrad(x)) {
    out.requiresGrad = true;
    record(() => {
      const dy = out.grad;
      if (!dy) return;
      const dx = x.g();
      for (let r = 0; r < R; r++) {
        const o = r * D;
        let dot = 0;
        for (let j = 0; j < D; j++) dot += dy[o + j] * y[o + j];
        for (let j = 0; j < D; j++) dx[o + j] += y[o + j] * (dy[o + j] - dot);
      }
    });
  }
  return out;
}

/** Sum of all elements (scalar). Mostly for tests and auxiliary losses. */
export function sum(x: Tensor): Tensor {
  let s = 0;
  for (let i = 0; i < x.size; i++) s += x.data[i];
  const out = Tensor.from([s], [1]);
  if (needsGrad(x)) {
    out.requiresGrad = true;
    record(() => {
      const dy = out.grad;
      if (!dy) return;
      const dx = x.g();
      const g = dy[0];
      for (let i = 0; i < dx.length; i++) dx[i] += g;
    });
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Normalisation
// ---------------------------------------------------------------------------------------------

/** LayerNorm over the last dim with gain g[D] and bias b[D]. Fused forward and backward. */
export function layernorm(x: Tensor, g: Tensor, b: Tensor, eps = 1e-5): Tensor {
  const D = lastDim(x);
  const R = rows(x);
  assert(g.size === D && b.size === D, "layernorm: gain/bias size mismatch");
  const out = Tensor.zeros(x.shape);
  const xd = x.data, y = out.data, gd = g.data, bd = b.data;
  const xhat = new Float32Array(x.size);
  const rstd = new Float32Array(R);
  for (let r = 0; r < R; r++) {
    const o = r * D;
    let mean = 0;
    for (let j = 0; j < D; j++) mean += xd[o + j];
    mean /= D;
    let v = 0;
    for (let j = 0; j < D; j++) {
      const d = xd[o + j] - mean;
      v += d * d;
    }
    const rs = 1 / Math.sqrt(v / D + eps);
    rstd[r] = rs;
    for (let j = 0; j < D; j++) {
      const h = (xd[o + j] - mean) * rs;
      xhat[o + j] = h;
      y[o + j] = h * gd[j] + bd[j];
    }
  }
  if (needsGrad(x, g, b)) {
    out.requiresGrad = true;
    record(() => {
      const dy = out.grad;
      if (!dy) return;
      const dg = g.requiresGrad ? g.g() : null;
      const db = b.requiresGrad ? b.g() : null;
      const dx = x.requiresGrad ? x.g() : null;
      for (let r = 0; r < R; r++) {
        const o = r * D;
        let sumDh = 0, sumDhH = 0;
        for (let j = 0; j < D; j++) {
          const d = dy[o + j];
          const h = xhat[o + j];
          if (dg) dg[j] += d * h;
          if (db) db[j] += d;
          const dh = d * gd[j];
          sumDh += dh;
          sumDhH += dh * h;
        }
        if (dx) {
          const rs = rstd[r];
          const m1 = sumDh / D, m2 = sumDhH / D;
          for (let j = 0; j < D; j++) {
            const dh = dy[o + j] * gd[j];
            dx[o + j] += rs * (dh - m1 - xhat[o + j] * m2);
          }
        }
      }
    });
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------------------------

/** Gather rows of table[V,D] for each id. Output shape [...idsShape, D]. Backward scatter-adds. */
export function embedding(ids: IntArray, table: Tensor, idsShape?: number[]): Tensor {
  const [V, D] = table.shape;
  const N = ids.length;
  const out = Tensor.zeros([...(idsShape ?? [N]), D]);
  const y = out.data, w = table.data;
  for (let i = 0; i < N; i++) {
    const id = ids[i];
    assert(id >= 0 && id < V, `embedding: id ${id} out of range [0,${V})`);
    y.set(w.subarray(id * D, id * D + D), i * D);
  }
  if (needsGrad(table)) {
    out.requiresGrad = true;
    record(() => {
      const dy = out.grad;
      if (!dy) return;
      const dw = table.g();
      for (let i = 0; i < N; i++) {
        const o = ids[i] * D, s = i * D;
        for (let j = 0; j < D; j++) dw[o + j] += dy[s + j];
      }
    });
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Causal multi-head self-attention (fused)
// ---------------------------------------------------------------------------------------------

export interface AttentionResult {
  out: Tensor;
  /** Attention probabilities, laid out [B, H, T, T] (row = query, col = key; zero above the diagonal). */
  probs: Float32Array;
}

/**
 * qkv: [B*T, 3*D] where columns [0,D) are queries, [D,2D) keys and [2D,3D) values; head h uses
 * columns h*hd..(h+1)*hd of each block. Returns softmax(QK^T/sqrt(hd) + causal mask) V, merged
 * back to [B*T, D].
 *
 * Per (sequence, head) the score and value products run through the GEMM kernel on contiguous
 * copies of Q, K, V. The kernel computes the full T x T score square (twice the causal work)
 * because the SIMD kernel is ~6x faster than a triangular scalar loop. The mask is applied in the
 * softmax: probabilities above the diagonal are exactly 0, so future positions cannot leak.
 */
export function causalAttention(qkv: Tensor, B: number, T: number, H: number): AttentionResult {
  const D3 = lastDim(qkv);
  const D = D3 / 3;
  assert(D3 % 3 === 0 && D % H === 0, "attention: bad qkv width / head count");
  assert(rows(qkv) === B * T, "attention: rows != B*T");
  const hd = D / H;
  const sc = 1 / Math.sqrt(hd);
  const x = qkv.data;
  const out = Tensor.zeros([B * T, D]);
  const y = out.data;
  const P = new Float32Array(B * H * T * T);
  const TT = T * T;
  const Q = new Float32Array(T * hd), Kh = new Float32Array(T * hd), Vt = new Float32Array(hd * T), O = new Float32Array(T * hd);

  const gather = (b: number, col: number, dst: Float32Array, transposed: boolean): void => {
    for (let t = 0; t < T; t++) {
      const src = (b * T + t) * D3 + col;
      if (transposed) for (let c = 0; c < hd; c++) dst[c * T + t] = x[src + c];
      else for (let c = 0; c < hd; c++) dst[t * hd + c] = x[src + c];
    }
  };

  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      gather(b, h * hd, Q, false);
      gather(b, D + h * hd, Kh, false);
      gather(b, 2 * D + h * hd, Vt, true);
      const pOff = (b * H + h) * TT;
      // S = Q K^T into the P slab, then a masked, scaled softmax row by row.
      gemmNT(T, T, hd, Q, 0, Kh, 0, P, pOff, false);
      for (let i = 0; i < T; i++) {
        const r = pOff + i * T;
        let m = -Infinity;
        for (let j = 0; j <= i; j++) {
          const v = P[r + j] * sc;
          P[r + j] = v;
          if (v > m) m = v;
        }
        let z = 0;
        for (let j = 0; j <= i; j++) {
          const e = Math.exp(P[r + j] - m);
          P[r + j] = e;
          z += e;
        }
        const inv = 1 / z;
        for (let j = 0; j <= i; j++) P[r + j] *= inv;
        for (let j = i + 1; j < T; j++) P[r + j] = 0;
      }
      // O = P V  (NT with V^T [hd, T])
      gemmNT(T, hd, T, P, pOff, Vt, 0, O, 0, false);
      for (let t = 0; t < T; t++) {
        const dst = (b * T + t) * D + h * hd;
        for (let c = 0; c < hd; c++) y[dst + c] = O[t * hd + c];
      }
    }
  }

  if (needsGrad(qkv)) {
    out.requiresGrad = true;
    record(() => {
      const dO = out.grad;
      if (!dO) return;
      const dx = qkv.g();
      const V = new Float32Array(T * hd), Qt = new Float32Array(hd * T), Kt = new Float32Array(hd * T);
      const dOh = new Float32Array(T * hd), dOt = new Float32Array(hd * T);
      const dS = new Float32Array(TT), tmpT = new Float32Array(TT);
      const dQ = new Float32Array(T * hd), dK = new Float32Array(T * hd), dV = new Float32Array(T * hd);
      for (let b = 0; b < B; b++) {
        for (let h = 0; h < H; h++) {
          gather(b, h * hd, Qt, true);
          gather(b, D + h * hd, Kt, true);
          gather(b, 2 * D + h * hd, V, false);
          for (let t = 0; t < T; t++) {
            const src = (b * T + t) * D + h * hd;
            for (let c = 0; c < hd; c++) {
              const g = dO[src + c];
              dOh[t * hd + c] = g;
              dOt[c * T + t] = g;
            }
          }
          const pOff = (b * H + h) * TT;
          // dP = dO V^T [T,T] (computed into dS, then turned into dS in place)
          gemmNT(T, T, hd, dOh, 0, V, 0, dS, 0, false);
          for (let i = 0; i < T; i++) {
            const r = i * T, pr = pOff + i * T;
            let dot = 0;
            for (let j = 0; j <= i; j++) dot += P[pr + j] * dS[r + j];
            for (let j = 0; j <= i; j++) dS[r + j] = P[pr + j] * (dS[r + j] - dot) * sc;
            for (let j = i + 1; j < T; j++) dS[r + j] = 0;
          }
          // dV = P^T dO : NT(P^T [T,T], dO^T [hd,T])
          transpose(P.subarray(pOff, pOff + TT), T, T, tmpT);
          gemmNT(T, hd, T, tmpT, 0, dOt, 0, dV, 0, false);
          // dQ = dS K : NT(dS, K^T)
          gemmNT(T, hd, T, dS, 0, Kt, 0, dQ, 0, false);
          // dK = dS^T Q : NT(dS^T, Q^T)
          transpose(dS, T, T, tmpT);
          gemmNT(T, hd, T, tmpT, 0, Qt, 0, dK, 0, false);
          for (let t = 0; t < T; t++) {
            const base = (b * T + t) * D3;
            const q = base + h * hd, k = base + D + h * hd, v = base + 2 * D + h * hd;
            for (let c = 0; c < hd; c++) {
              dx[q + c] += dQ[t * hd + c];
              dx[k + c] += dK[t * hd + c];
              dx[v + c] += dV[t * hd + c];
            }
          }
        }
      }
    });
  }
  return { out, probs: P };
}

// ---------------------------------------------------------------------------------------------
// Loss
// ---------------------------------------------------------------------------------------------

/**
 * Mean cross-entropy between logits[N,V] and integer targets[N]. Fused log-softmax + NLL.
 * Targets < 0 are ignored (not counted in the mean).
 */
export function crossEntropy(logits: Tensor, targets: IntArray): Tensor {
  const V = lastDim(logits);
  const N = rows(logits);
  assert(targets.length === N, `crossEntropy: ${targets.length} targets for ${N} rows`);
  const z = logits.data;
  const probs = new Float32Array(N * V);
  let loss = 0;
  let count = 0;
  for (let r = 0; r < N; r++) {
    const o = r * V;
    let m = -Infinity;
    for (let j = 0; j < V; j++) if (z[o + j] > m) m = z[o + j];
    let s = 0;
    for (let j = 0; j < V; j++) {
      const e = Math.exp(z[o + j] - m);
      probs[o + j] = e;
      s += e;
    }
    const inv = 1 / s;
    for (let j = 0; j < V; j++) probs[o + j] *= inv;
    const t = targets[r];
    if (t >= 0) {
      loss += -(z[o + t] - m - Math.log(s));
      count++;
    }
  }
  const denom = Math.max(count, 1);
  const out = Tensor.from([loss / denom], [1]);
  if (needsGrad(logits)) {
    out.requiresGrad = true;
    record(() => {
      const dy = out.grad;
      if (!dy) return;
      const g = dy[0] / denom;
      const dz = logits.g();
      for (let r = 0; r < N; r++) {
        const t = targets[r];
        if (t < 0) continue;
        const o = r * V;
        for (let j = 0; j < V; j++) dz[o + j] += g * probs[o + j];
        dz[o + t] -= g;
      }
    });
  }
  return out;
}
