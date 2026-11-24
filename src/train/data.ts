/**
 * Token shards and batch sampling.
 *
 * Shard format (written by the tokenizer step): data/train.bin and data/val.bin are raw
 * little-endian uint16 token ids, files concatenated with the <|bof|>/<|eof|> specials between
 * them. Shards are loaded into a SharedArrayBuffer so every worker thread reads the same memory.
 * Training batches are random windows (offset uniform over the shard); validation uses a fixed,
 * evenly spaced set of non-overlapping windows so every eval sees exactly the same tokens.
 */
import { existsSync, readFileSync } from "node:fs";
import { Rng } from "../engine/rng.js";

export function loadTokenShard(path: string): Uint16Array {
  const buf = readFileSync(path);
  if (buf.byteLength % 2 !== 0) throw new Error(`${path}: odd byte length, expected uint16 tokens`);
  const out = new Uint16Array(new SharedArrayBuffer(buf.byteLength));
  new Uint8Array(out.buffer).set(buf);
  return out;
}

export function maxToken(tokens: Uint16Array): number {
  let m = 0;
  for (let i = 0; i < tokens.length; i++) if (tokens[i] > m) m = tokens[i];
  return m;
}

/** Optional data/meta.json written by the tokenizer step: { vocabSize } or { vocab_size }. */
export function readVocabFromMeta(dataDir: string): number | null {
  const p = `${dataDir}/meta.json`;
  if (!existsSync(p)) return null;
  const meta = JSON.parse(readFileSync(p, "utf8")) as Record<string, unknown>;
  const v = meta.vocabSize ?? meta.vocab_size;
  return typeof v === "number" ? v : null;
}

/** `count` random window starts such that tokens[off .. off+T] (T+1 tokens) is in range. */
export function sampleOffsets(rng: Rng, nTokens: number, T: number, count: number): number[] {
  const hi = nTokens - T - 1;
  if (hi <= 0) throw new Error(`shard of ${nTokens} tokens is too short for windows of ${T}`);
  return Array.from({ length: count }, () => rng.int(hi + 1));
}

/** Evenly spaced, non-overlapping validation windows covering at most maxTokens tokens. */
export function evalOffsets(nTokens: number, T: number, maxTokens: number): number[] {
  const nWin = Math.max(1, Math.min(Math.floor((nTokens - 1) / T), Math.floor(maxTokens / T)));
  const stride = Math.floor((nTokens - T - 1) / Math.max(1, nWin - 1)) || T;
  const out: number[] = [];
  for (let i = 0; i < nWin; i++) out.push(Math.min(i * Math.max(stride, T), nTokens - T - 1));
  return out;
}

/** Build inputs [B*T] and next-token targets [B*T] from window starts. */
export function buildBatch(tokens: Uint16Array, offsets: number[], T: number): { ids: Int32Array; targets: Int32Array } {
  const B = offsets.length;
  const ids = new Int32Array(B * T);
  const targets = new Int32Array(B * T);
  for (let b = 0; b < B; b++) {
    const o = offsets[b];
    for (let t = 0; t < T; t++) {
      ids[b * T + t] = tokens[o + t];
      targets[b * T + t] = tokens[o + t + 1];
    }
  }
  return { ids, targets };
}

/**
 * Synthetic token stream for smoke tests and benchmarks when no corpus exists yet: a sparse
 * random 2nd-order Markov chain (each context has 4 likely successors), so a model can learn it
 * well below the uniform loss but not to zero.
 */
export function syntheticTokens(n: number, vocab: number, seed = 1): Uint16Array {
  const rng = new Rng(seed);
  const out = new Uint16Array(new SharedArrayBuffer(n * 2));
  const succ = new Uint16Array(vocab * 4);
  for (let i = 0; i < succ.length; i++) succ[i] = rng.int(vocab);
  let a = 0, b = 1;
  for (let i = 0; i < n; i++) {
    const r = rng.next();
    const c = r < 0.05 ? rng.int(vocab) : succ[((a * 31 + b) % vocab) * 4 + Math.floor(rng.next() * 4)];
    out[i] = c;
    a = b;
    b = c;
  }
  return out;
}
