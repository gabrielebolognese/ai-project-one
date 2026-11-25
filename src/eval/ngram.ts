/**
 * Token n-gram baseline: interpolated modified Kneser-Ney (Chen & Goodman 1998), built with typed
 * arrays only so a 5-gram over the 24M-token train split fits in a few hundred MB.
 *
 * Construction (no hashing, no string keys):
 *   - order N: LSD radix sort of every stream position by its N following tokens (each digit is a
 *     token id, 2048 buckets), then one scan to collapse equal n-grams into (tokens, raw count);
 *   - order n < N: the continuation count N1+(. w1..wn) is the number of distinct (n+1)-grams
 *     ending in w1..wn, so the unique (n+1)-gram list is radix-sorted on its last n columns and
 *     run-length counted.
 * Each table is sorted lexicographically, so all continuations of a context are one contiguous
 * range found by two binary searches.
 *
 * Prediction returns the full distribution over the vocabulary (needed for top-k and perplexity):
 *   p_0 = uniform;  p_n(w|c) = max(c(cw) - D_n(c(cw)), 0) / c(c.) + gamma_n(c) * p_{n-1}(w|c'),
 *   gamma_n(c) = (D1 N1(c.) + D2 N2(c.) + D3 N3+(c.)) / c(c.),
 * where counts are raw at the top order and continuation counts below it. Orders whose context
 * was never seen are skipped (p_n = p_{n-1}).
 */

export interface NgramTable {
  n: number;
  /** Unique n-grams, row-major [U * n], sorted lexicographically. */
  grams: Uint16Array;
  /** Raw count (top order) or continuation count (lower orders). */
  counts: Uint32Array;
  /** Modified-KN discounts for counts 1, 2, 3+. */
  D: [number, number, number];
}

export interface NgramModel {
  order: number;
  vocab: number;
  tables: NgramTable[]; // tables[n-1] is order n
  /** Precomputed order-1 distribution (context-free). */
  p1: Float64Array;
}

/** Stable counting sort of `perm` by key(perm[i]) in [0, V). */
function countingSort(perm: Uint32Array, tmp: Uint32Array, key: (i: number) => number, V: number): void {
  const cnt = new Uint32Array(V + 1);
  const n = perm.length;
  for (let i = 0; i < n; i++) cnt[key(perm[i]) + 1]++;
  for (let v = 0; v < V; v++) cnt[v + 1] += cnt[v];
  for (let i = 0; i < n; i++) {
    const p = perm[i];
    tmp[cnt[key(p)]++] = p;
  }
  perm.set(tmp);
}

function discounts(counts: Uint32Array): [number, number, number] {
  let n1 = 0, n2 = 0, n3 = 0, n4 = 0;
  for (let i = 0; i < counts.length; i++) {
    const c = counts[i];
    if (c === 1) n1++;
    else if (c === 2) n2++;
    else if (c === 3) n3++;
    else if (c === 4) n4++;
  }
  if (n1 === 0 || n2 === 0 || n3 === 0) return [0.5, 0.75, 0.9];
  const Y = n1 / (n1 + 2 * n2);
  const clamp = (d: number, hi: number): number => Math.min(hi, Math.max(0.05, d));
  return [clamp(1 - 2 * Y * (n2 / n1), 1), clamp(2 - 3 * Y * (n3 / n2), 2), clamp(3 - 4 * Y * (n4 / n3), 3)];
}

/** Collapse a lexicographically sorted list of rows (given by an accessor) into unique rows + counts. */
function collapse(
  m: number,
  n: number,
  tok: (row: number, col: number) => number,
): { grams: Uint16Array; counts: Uint32Array } {
  // First pass: count uniques.
  let U = 0;
  for (let i = 0; i < m; i++) {
    let same = i > 0;
    if (same) for (let c = 0; c < n; c++) if (tok(i, c) !== tok(i - 1, c)) { same = false; break; }
    if (!same) U++;
  }
  const grams = new Uint16Array(U * n);
  const counts = new Uint32Array(U);
  let u = -1;
  for (let i = 0; i < m; i++) {
    let same = i > 0;
    if (same) for (let c = 0; c < n; c++) if (tok(i, c) !== tok(i - 1, c)) { same = false; break; }
    if (!same) {
      u++;
      for (let c = 0; c < n; c++) grams[u * n + c] = tok(i, c);
    }
    counts[u]++;
  }
  return { grams, counts };
}

export function buildNgram(stream: Uint16Array, vocab: number, order: number, log?: (m: string) => void): NgramModel {
  if (order < 1 || order > 8) throw new Error("order must be in 1..8");
  const tables: NgramTable[] = new Array(order);
  // Top order: radix sort stream positions.
  const m = stream.length - order + 1;
  let perm = new Uint32Array(m);
  for (let i = 0; i < m; i++) perm[i] = i;
  let tmp = new Uint32Array(m);
  for (let j = order - 1; j >= 0; j--) countingSort(perm, tmp, (p) => stream[p + j], vocab);
  {
    const p = perm;
    const top = collapse(m, order, (row, col) => stream[p[row] + col]);
    tables[order - 1] = { n: order, ...top, D: discounts(top.counts) };
    log?.(`[ngram] order ${order}: ${top.counts.length.toLocaleString()} unique of ${m.toLocaleString()}`);
  }
  // Lower orders: continuation counts from the unique (n+1)-grams.
  for (let n = order - 1; n >= 1; n--) {
    const up = tables[n];
    const g = up.grams, w = n + 1, U = up.counts.length;
    perm = new Uint32Array(U);
    for (let i = 0; i < U; i++) perm[i] = i;
    tmp = new Uint32Array(U);
    for (let j = n; j >= 1; j--) countingSort(perm, tmp, (r) => g[r * w + j], vocab);
    const p = perm;
    const t = collapse(U, n, (row, col) => g[p[row] * w + col + 1]);
    tables[n - 1] = { n, ...t, D: discounts(t.counts) };
    log?.(`[ngram] order ${n}: ${t.counts.length.toLocaleString()} unique (continuation counts)`);
  }
  // Order-1 distribution over a uniform base.
  const p1 = new Float64Array(vocab).fill(1 / vocab);
  applyOrder(tables[0], 0, tables[0].counts.length, p1);
  return { order, vocab, tables, p1 };
}

/** Interpolate p (in place) with the continuation range [lo, hi) of a table. */
function applyOrder(t: NgramTable, lo: number, hi: number, p: Float64Array): void {
  if (hi <= lo) return;
  const { counts, grams, n, D } = t;
  let total = 0, n1 = 0, n2 = 0, n3 = 0;
  for (let i = lo; i < hi; i++) {
    const c = counts[i];
    total += c;
    if (c === 1) n1++;
    else if (c === 2) n2++;
    else n3++;
  }
  const gamma = (D[0] * n1 + D[1] * n2 + D[2] * n3) / total;
  for (let v = 0; v < p.length; v++) p[v] *= gamma;
  for (let i = lo; i < hi; i++) {
    const c = counts[i];
    const d = c === 1 ? D[0] : c === 2 ? D[1] : D[2];
    p[grams[i * n + n - 1]] += Math.max(c - d, 0) / total;
  }
}

/** [lo, hi) of rows in table t whose first n-1 tokens equal ctx (ctx.length === t.n - 1). */
function contextRange(t: NgramTable, ctx: ArrayLike<number>, ctxStart: number): [number, number] {
  const { grams, n } = t;
  const k = n - 1;
  const U = t.counts.length;
  const cmp = (row: number): number => {
    const o = row * n;
    for (let c = 0; c < k; c++) {
      const d = grams[o + c] - ctx[ctxStart + c];
      if (d !== 0) return d;
    }
    return 0;
  };
  let lo = 0, hi = U;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (cmp(mid) < 0) lo = mid + 1;
    else hi = mid;
  }
  const first = lo;
  hi = U;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (cmp(mid) <= 0) lo = mid + 1;
    else hi = mid;
  }
  return [first, lo];
}

/**
 * Next-token distribution given the history `hist[0..end)` (only the last order-1 tokens are used).
 * Writes into `out` (length vocab) and returns it.
 */
export function ngramDist(m: NgramModel, hist: ArrayLike<number>, end: number, out = new Float64Array(m.vocab)): Float64Array {
  out.set(m.p1);
  for (let n = 2; n <= m.order; n++) {
    const k = n - 1;
    if (end - k < 0) break;
    const [lo, hi] = contextRange(m.tables[n - 1], hist, end - k);
    if (hi <= lo) continue; // unseen context: keep the lower-order distribution
    applyOrder(m.tables[n - 1], lo, hi, out);
  }
  return out;
}

export function ngramSizeBytes(m: NgramModel): number {
  return m.tables.reduce((s, t) => s + t.grams.byteLength + t.counts.byteLength, 0);
}

/**
 * Cache component (Tu, Su & Devanbu 2014; Hellendoorn & Devanbu 2017): code is locally repetitive,
 * so the tokens that followed the current context earlier in the same file are good guesses. Finds
 * the longest suffix of the history (up to maxK tokens) that occurred earlier in
 * hist[from..end), and returns the empirical next-token counts after it.
 * Returns the total count (0 = no match); `counts` is zeroed for the touched ids only via `touched`.
 */
export function cacheCounts(
  hist: ArrayLike<number>,
  from: number,
  end: number,
  maxK: number,
  counts: Float64Array,
  touched: number[],
): number {
  for (const t of touched) counts[t] = 0;
  touched.length = 0;
  for (let k = Math.min(maxK, end - from - 1); k >= 1; k--) {
    let total = 0;
    // occurrences of hist[end-k..end) ending before position j, followed by hist[j]
    for (let j = from + k; j < end; j++) {
      let ok = true;
      for (let c = 1; c <= k; c++) if (hist[j - c] !== hist[end - c]) { ok = false; break; }
      if (!ok) continue;
      const w = hist[j];
      if (counts[w] === 0) touched.push(w);
      counts[w]++;
      total++;
    }
    if (total > 0) return total;
  }
  return 0;
}

/** p <- (1 - lam) p + lam * cache, lam = total / (total + beta). */
export function mixCache(p: Float64Array, counts: Float64Array, touched: number[], total: number, beta: number): void {
  if (total <= 0) return;
  const lam = total / (total + beta);
  for (let v = 0; v < p.length; v++) p[v] *= 1 - lam;
  for (const w of touched) p[w] += (lam * counts[w]) / total;
}
