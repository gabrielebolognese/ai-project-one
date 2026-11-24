/** Tiny PCA: covariance + power iteration with deflation. Deterministic (fixed start vectors). */
export interface PcaBasis {
  d: number;
  k: number;
  /** Rows used for the fit. */
  n: number;
  mean: Float32Array;
  /** k x d, unit-norm principal directions (sign fixed so the largest |entry| is positive). */
  components: Float32Array;
  /** Fraction of total variance explained by each component. */
  explained: number[];
}

export function fitPca(rows: Float32Array[], d: number, k: number): PcaBasis {
  const n = rows.length;
  const mean = new Float64Array(d);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j];
  for (let j = 0; j < d; j++) mean[j] /= Math.max(1, n);
  const cov = new Float64Array(d * d);
  const c = new Float64Array(d);
  for (const r of rows) {
    for (let j = 0; j < d; j++) c[j] = r[j] - mean[j];
    for (let i = 0; i < d; i++) {
      const ci = c[i];
      if (ci === 0) continue;
      const o = i * d;
      for (let j = i; j < d; j++) cov[o + j] += ci * c[j];
    }
  }
  for (let i = 0; i < d; i++) for (let j = i; j < d; j++) {
    const v = cov[i * d + j] / Math.max(1, n - 1);
    cov[i * d + j] = v;
    cov[j * d + i] = v;
  }
  let total = 0;
  for (let i = 0; i < d; i++) total += cov[i * d + i];
  const comps = new Float32Array(k * d);
  const explained: number[] = [];
  const v = new Float64Array(d);
  const w = new Float64Array(d);
  for (let p = 0; p < k; p++) {
    for (let j = 0; j < d; j++) v[j] = 1 + ((j * 7919 + p * 104729) % 13) / 13;
    let lambda = 0;
    for (let it = 0; it < 300; it++) {
      for (let i = 0; i < d; i++) {
        let s = 0;
        const o = i * d;
        for (let j = 0; j < d; j++) s += cov[o + j] * v[j];
        w[i] = s;
      }
      let norm = 0;
      for (let i = 0; i < d; i++) norm += w[i] * w[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-20) break;
      let delta = 0;
      for (let i = 0; i < d; i++) {
        const nv = w[i] / norm;
        delta += Math.abs(nv - v[i]);
        v[i] = nv;
      }
      lambda = norm;
      if (delta < 1e-9) break;
    }
    let big = 0;
    for (let j = 0; j < d; j++) if (Math.abs(v[j]) > Math.abs(big)) big = v[j];
    const sgn = big < 0 ? -1 : 1;
    for (let j = 0; j < d; j++) comps[p * d + j] = sgn * v[j];
    explained.push(total > 0 ? lambda / total : 0);
    // deflate
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i * d + j] -= lambda * v[i] * v[j];
  }
  return { d, k, n, mean: Float32Array.from(mean), components: comps, explained };
}

/** Project T rows of a [T, d] matrix into k dims. Returns [T * k]. */
export function project(basis: PcaBasis, mat: Float32Array, T: number): Float32Array {
  const { d, k, mean, components } = basis;
  const out = new Float32Array(T * k);
  for (let t = 0; t < T; t++) {
    const o = t * d;
    for (let p = 0; p < k; p++) {
      let s = 0;
      const co = p * d;
      for (let j = 0; j < d; j++) s += (mat[o + j] - mean[j]) * components[co + j];
      out[t * k + p] = s;
    }
  }
  return out;
}
