/**
 * Inference + compact activation tracing for the lab.
 *
 * The trace comes from the same Transformer.forward the trainer uses. Generation uses the KV-cached
 * IncrementalDecoder (decoder.ts), built from the same engine ops and tested against forward().
 * Timings are measured by `npm run smoke`.
 *
 * Trace payloads are deliberately compact for keystroke-rate streaming:
 *   - residual stream: 3-D PCA points per token per stage (fixed basis) + L2 norms; the full
 *     d-dim vectors only for the last token,
 *   - attention: every head's row for the last query over all keys, plus the full matrix for the
 *     last `attnWindow` tokens,
 *   - MLP: top-k neurons of the last token per layer, plus the argmax neuron per token,
 *   - numbers rounded to 3-4 significant decimals.
 */
import { noGrad } from "../engine/index.js";
import { BOF, EOF } from "../tokenizer/index.js";
import type { ModelHost } from "./model-host.js";
import { project } from "./pca.js";
import { IncrementalDecoder } from "./decoder.js";

export interface TokenView {
  id: number;
  text: string;
}

export interface TopK {
  id: number;
  text: string;
  p: number;
  logit: number;
}

export interface StepInfo {
  i: number;
  id: number;
  text: string;
  p: number;
  entropy: number;
  topk: TopK[];
}

export interface Prompt {
  ids: number[];
  tokens: TokenView[];
  /** Number of leading tokens dropped to fit the context window. */
  truncated: number;
  prefixChars: number;
}

export interface GenerateOptions {
  maxTokens?: number;
  topK?: number;
  /** 0 = greedy (default). */
  temperature?: number;
  /** Stop at the first newline after some non-whitespace output (line completion). Default true. */
  stopAtNewline?: boolean;
  seed?: number;
}

export const r3 = (x: number): number => Math.round(x * 1000) / 1000;
export const r4 = (x: number): number => Math.round(x * 10000) / 10000;

export function tokenText(host: ModelHost, id: number): string {
  if (id === BOF) return "<|bof|>";
  if (id === EOF) return "<|eof|>";
  if (id === 2) return "<|pad|>";
  return host.tokenizer.decode([id]);
}

export function buildPrompt(host: ModelHost, code: string, cursor?: number): Prompt {
  const c = cursor === undefined || cursor === null ? code.length : Math.max(0, Math.min(code.length, Math.floor(cursor)));
  const prefix = code.slice(0, c);
  let ids = [BOF, ...host.tokenizer.encode(prefix)];
  const ctx = host.model.cfg.ctx;
  let truncated = 0;
  if (ids.length > ctx) {
    truncated = ids.length - ctx;
    ids = ids.slice(truncated);
  }
  return { ids, tokens: ids.map((id) => ({ id, text: tokenText(host, id) })), truncated, prefixChars: c };
}

/** Softmax over one logits row (optionally tempered) and the top-k entries. */
export function topKOf(host: ModelHost, row: Float32Array, k: number, temperature = 1): { topk: TopK[]; probs: Float32Array; entropy: number } {
  const V = row.length;
  const t = temperature > 0 ? temperature : 1;
  let mx = -Infinity;
  for (let i = 0; i < V; i++) if (row[i] > mx) mx = row[i];
  const probs = new Float32Array(V);
  let s = 0;
  for (let i = 0; i < V; i++) {
    const e = Math.exp((row[i] - mx) / t);
    probs[i] = e;
    s += e;
  }
  let ent = 0;
  for (let i = 0; i < V; i++) {
    const p = probs[i] / s;
    probs[i] = p;
    if (p > 0) ent -= p * Math.log(p);
  }
  // partial selection of the k largest
  const idx: number[] = [];
  for (let i = 0; i < V; i++) {
    if (idx.length < k) {
      idx.push(i);
      if (idx.length === k) idx.sort((a, b) => probs[b] - probs[a]);
    } else if (probs[i] > probs[idx[k - 1]]) {
      let j = k - 1;
      while (j > 0 && probs[idx[j - 1]] < probs[i]) { idx[j] = idx[j - 1]; j--; }
      idx[j] = i;
    }
  }
  if (idx.length < k) idx.sort((a, b) => probs[b] - probs[a]);
  return {
    topk: idx.map((id) => ({ id, text: tokenText(host, id), p: r4(probs[id]), logit: r3(row[id]) })),
    probs,
    entropy: r3(ent),
  };
}

/** Deterministic xorshift for optional sampling. */
function makeRng(seed: number): () => number {
  let x = (seed >>> 0) || 0x9e3779b9;
  return () => {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    return x / 4294967296;
  };
}

const yieldToLoop = () => new Promise<void>((r) => setImmediate(r));

/**
 * Autoregressive generation. `onStep` is called after every token; `isStale` is polled between
 * steps (after yielding to the event loop, so newer WS messages get a chance to arrive) and stops
 * generation early. Returns the generated steps and whether it was cancelled.
 */
export async function generate(
  host: ModelHost,
  promptIds: number[],
  opts: GenerateOptions = {},
  onStep?: (s: StepInfo) => void,
  isStale?: () => boolean,
): Promise<{ steps: StepInfo[]; text: string; cancelled: boolean; stopReason: string }> {
  const maxTokens = Math.max(0, Math.min(128, Math.floor(opts.maxTokens ?? 16)));
  const k = Math.max(1, Math.min(50, Math.floor(opts.topK ?? 10)));
  const temperature = opts.temperature ?? 0;
  const stopAtNewline = opts.stopAtNewline ?? true;
  const rand = makeRng(opts.seed ?? 42);
  const ctx = host.model.cfg.ctx;
  const slide = Math.max(1, Math.min(32, ctx >> 2));
  const dec = new IncrementalDecoder(host.model);
  let ids = promptIds.slice(-ctx);
  let row = dec.prefill(ids);
  const steps: StepInfo[] = [];
  const outIds: number[] = [];
  let stopReason = "maxTokens";
  for (let i = 0; i < maxTokens; i++) {
    if (isStale?.()) return { steps, text: host.tokenizer.decode(outIds), cancelled: true, stopReason: "cancelled" };
    const { topk, probs, entropy } = topKOf(host, row, k, temperature > 0 ? temperature : 1);
    let id = topk[0].id;
    if (temperature > 0) {
      let u = rand();
      for (let j = 0; j < probs.length; j++) {
        u -= probs[j];
        if (u <= 0) { id = j; break; }
      }
    }
    const step: StepInfo = { i, id, text: tokenText(host, id), p: r4(probs[id]), entropy, topk };
    steps.push(step);
    onStep?.(step);
    if (id === EOF) { stopReason = "eof"; break; }
    outIds.push(id);
    ids.push(id);
    if (stopAtNewline) {
      const txt = host.tokenizer.decode(outIds);
      const nl = txt.indexOf("\n");
      if (nl >= 0 && txt.slice(0, nl).trim().length > 0) { stopReason = "newline"; break; }
    }
    if (i + 1 >= maxTokens) break;
    if (dec.full) {
      // context shift: rebuild the cache on the most recent ctx - slide tokens
      ids = ids.slice(-(ctx - slide));
      row = dec.prefill(ids);
    } else {
      row = dec.step(id);
    }
    await yieldToLoop();
  }
  let text = host.tokenizer.decode(outIds);
  if (stopAtNewline && stopReason === "newline") text = text.slice(0, text.indexOf("\n"));
  return { steps, text, cancelled: false, stopReason };
}

export interface TraceOptions {
  topK?: number;
  /** Full attention matrices are sent for the last `attnWindow` tokens (default 16, max 64). */
  attnWindow?: number;
  /** Top MLP neurons per layer for the last token (default 12). */
  mlpTopK?: number;
  /** Include full d-dim residual vectors for the last token (default true). */
  lastVectors?: boolean;
  /** Include every MLP activation of the last token (default true; dMlp floats per layer). */
  mlpLastAll?: boolean;
  /** Top neurons per window token per layer (default 8, max 32). */
  mlpWindowTopK?: number;
}

export interface TracePayload {
  T: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  dMlp: number;
  /** Stage names in order: embed, L0.attn, L0.mlp, L1.attn, ... */
  stages: string[];
  /** Per stage: [T*3] PCA coordinates (row-major by token). */
  pca: number[][];
  /** Per stage: [T] residual L2 norms. */
  norms: number[][];
  /** Per stage: full residual vector of the last token [d]. */
  lastVectors?: number[][];
  attention: {
    /** [L][H][T]: attention of the last query position over all keys. */
    last: number[][][];
    /** First token index of the window. */
    windowStart: number;
    window: number;
    /** [L][H][W*W]: probabilities for queries and keys inside the window (row-major, query rows). */
    matrices: number[][][];
  };
  mlp: {
    /** [L]: top neurons for the last token. */
    lastTop: { i: number; v: number }[][];
    /** [L][T]: argmax neuron index per token. */
    argmax: number[][];
    /** [L][T]: argmax neuron value per token. */
    argmaxV: number[][];
    /** [L]: fraction of the last token's neurons with activation > 0. */
    activeFrac: number[];
    /** [L][dMlp]: every post-GELU activation of the last token (for the neuron grid). */
    lastAll?: number[][];
    /** [L][W][k]: top neurons of each token inside the attention window (token-by-token replay). */
    windowTop: { i: number; v: number }[][][];
  };
  /** Logit lens [stage]: LN_f(residual of the last token) @ tok_emb^T, top 5 tokens per stage. */
  logitLens: { topk: TopK[]; entropy: number }[];
  next: { topk: TopK[]; entropy: number };
}

/** Decode a residual vector through the final LayerNorm and the tied head (logit lens). */
function lensRow(host: ModelHost, vec: Float32Array): Float32Array {
  const { dModel: d, vocabSize: V } = host.model.cfg;
  const s = host.model.store;
  const g = s.get("ln_f.g").data, b = s.get("ln_f.b").data, E = s.get("tok_emb").data;
  let mean = 0;
  for (let j = 0; j < d; j++) mean += vec[j];
  mean /= d;
  let v = 0;
  for (let j = 0; j < d; j++) v += (vec[j] - mean) ** 2;
  const rs = 1 / Math.sqrt(v / d + 1e-5);
  const f = new Float32Array(d);
  for (let j = 0; j < d; j++) f[j] = (vec[j] - mean) * rs * g[j] + b[j];
  const out = new Float32Array(V);
  for (let t = 0; t < V; t++) {
    let acc = 0;
    const o = t * d;
    for (let j = 0; j < d; j++) acc += f[j] * E[o + j];
    out[t] = acc;
  }
  return out;
}

/** One traced forward pass over the prompt window, condensed for the browser. */
export function tracePrompt(host: ModelHost, ids: number[], opts: TraceOptions = {}): TracePayload {
  const { dModel: d, nLayers: L, nHeads: H, dMlp: m, vocabSize: V } = host.model.cfg;
  const T = ids.length;
  const tr = noGrad(() => host.model.forward(ids, 1, T, { trace: true }).trace!);
  const stageMats: Float32Array[] = [tr.embed];
  const stages = ["embed"];
  tr.layers.forEach((l, i) => {
    stageMats.push(l.residMid, l.residOut);
    stages.push(`L${i}.attn`, `L${i}.mlp`);
  });
  const pca: number[][] = [];
  const norms: number[][] = [];
  const lastVectors: number[][] = [];
  for (const mat of stageMats) {
    pca.push(Array.from(project(host.pca, mat, T), r3));
    const nr: number[] = [];
    for (let t = 0; t < T; t++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += mat[t * d + j] * mat[t * d + j];
      nr.push(r3(Math.sqrt(s)));
    }
    norms.push(nr);
    if (opts.lastVectors !== false) lastVectors.push(Array.from(mat.subarray((T - 1) * d, T * d), r3));
  }

  const W = Math.min(T, Math.max(1, Math.min(64, Math.floor(opts.attnWindow ?? 16))));
  const w0 = T - W;
  const attLast: number[][][] = [];
  const attMats: number[][][] = [];
  for (const l of tr.layers) {
    const pr = l.attnProbs; // [1, H, T, T]
    const lastL: number[][] = [];
    const matsL: number[][] = [];
    for (let h = 0; h < H; h++) {
      const base = h * T * T;
      lastL.push(Array.from(pr.subarray(base + (T - 1) * T, base + T * T), r3));
      const mat: number[] = new Array(W * W);
      for (let qi = 0; qi < W; qi++) {
        const q = w0 + qi;
        for (let ki = 0; ki < W; ki++) mat[qi * W + ki] = r3(pr[base + q * T + w0 + ki]);
      }
      matsL.push(mat);
    }
    attLast.push(lastL);
    attMats.push(matsL);
  }

  const mk = Math.max(1, Math.min(64, Math.floor(opts.mlpTopK ?? 12)));
  const lastTop: { i: number; v: number }[][] = [];
  const argmax: number[][] = [];
  const argmaxV: number[][] = [];
  const activeFrac: number[] = [];
  const lastAll: number[][] = [];
  const windowTop: { i: number; v: number }[][][] = [];
  const wk = Math.max(1, Math.min(32, Math.floor(opts.mlpWindowTopK ?? 8)));
  for (const l of tr.layers) {
    const a = l.mlpAct; // [T, m]
    const row = a.subarray((T - 1) * m, T * m);
    const order = Array.from({ length: m }, (_, i) => i).sort((x, y) => row[y] - row[x]).slice(0, mk);
    lastTop.push(order.map((i) => ({ i, v: r3(row[i]) })));
    let act = 0;
    for (let i = 0; i < m; i++) if (row[i] > 0) act++;
    activeFrac.push(r3(act / m));
    const am: number[] = [];
    const av: number[] = [];
    for (let t = 0; t < T; t++) {
      let bi = 0, bv = -Infinity;
      for (let i = 0; i < m; i++) { const v = a[t * m + i]; if (v > bv) { bv = v; bi = i; } }
      am.push(bi);
      av.push(r3(bv));
    }
    argmax.push(am);
    argmaxV.push(av);
    if (opts.mlpLastAll !== false) lastAll.push(Array.from(row, r3));
    const wt: { i: number; v: number }[][] = [];
    for (let t = w0; t < T; t++) {
      const r = a.subarray(t * m, (t + 1) * m);
      const top: number[] = [];
      for (let i = 0; i < m; i++) {
        if (top.length < wk || r[i] > r[top[top.length - 1]]) {
          let j = top.length < wk ? top.length : wk - 1;
          if (top.length < wk) top.push(i);
          while (j > 0 && r[top[j - 1]] < r[i]) { top[j] = top[j - 1]; j--; }
          top[j] = i;
        }
      }
      wt.push(top.map((i) => ({ i, v: r3(r[i]) })));
    }
    windowTop.push(wt);
  }

  const logitLens = stageMats.map((mat) => {
    const r = topKOf(host, lensRow(host, mat.subarray((T - 1) * d, T * d)), 5);
    return { topk: r.topk, entropy: r.entropy };
  });
  const next = topKOf(host, tr.logits.subarray((T - 1) * V, T * V), Math.max(1, Math.min(50, opts.topK ?? 10)));
  return {
    T, dModel: d, nLayers: L, nHeads: H, dMlp: m, stages, pca, norms,
    ...(opts.lastVectors !== false ? { lastVectors } : {}),
    attention: { last: attLast, windowStart: w0, window: W, matrices: attMats },
    mlp: { lastTop, argmax, argmaxV, activeFrac, ...(opts.mlpLastAll !== false ? { lastAll } : {}), windowTop },
    logitLens,
    next: { topk: next.topk, entropy: next.entropy },
  };
}
