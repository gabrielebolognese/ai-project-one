/**
 * Byte-level BPE (Sennrich et al. 2015, with the byte alphabet of GPT-2) trained and run in plain
 * TypeScript. No node-only APIs, so the browser lab uses the very same encoder/decoder.
 *
 * Id layout:   0 <|bof|>   1 <|eof|>   2 <|pad|>   3..258 raw bytes 0..255   259.. merges in order
 */
import { makePretokenRegex, PRETOKEN_FLAGS, PRETOKEN_SOURCE } from "./pretokenize.js";

export const BOF = 0;
export const EOF = 1;
export const PAD = 2;
export const N_SPECIAL = 3;
export const BYTE_BASE = 3;
export const MERGE_BASE = 259;
export const SPECIAL_NAMES = ["<|bof|>", "<|eof|>", "<|pad|>"];

export interface TokenizerData {
  version: 1;
  type: "byte-level-bpe";
  vocabSize: number;
  specials: { bof: number; eof: number; pad: number };
  pretoken: { source: string; flags: string };
  /** merges[i] = [left, right] -> new id MERGE_BASE + i */
  merges: [number, number][];
  /** Human-readable form of every id (partial UTF-8 shown as U+FFFD; control chars escaped). */
  tokens: string[];
  trainInfo?: Record<string, unknown>;
}

const enc = new TextEncoder();
const dec = new TextDecoder("utf-8");

/** Count pre-tokens over an iterable of texts (the unit BPE training works on). */
export function countPretokens(texts: Iterable<string>, counts = new Map<string, number>()): Map<string, number> {
  const re = makePretokenRegex();
  for (const t of texts) {
    re.lastIndex = 0;
    for (const m of t.matchAll(re)) counts.set(m[0], (counts.get(m[0]) ?? 0) + 1);
  }
  return counts;
}

function displayToken(bytes: Uint8Array): string {
  return dec.decode(bytes).replace(/[\u0000-\u001f\u007f]/g, (c) => {
    if (c === "\n") return "\\n";
    if (c === "\t") return "\\t";
    if (c === "\r") return "\\r";
    return "\\x" + c.charCodeAt(0).toString(16).padStart(2, "0");
  });
}

function buildTokens(merges: [number, number][]): { bytes: Uint8Array[]; tokens: string[] } {
  const bytes: Uint8Array[] = [];
  for (let i = 0; i < N_SPECIAL; i++) bytes.push(new Uint8Array(0));
  for (let b = 0; b < 256; b++) bytes.push(Uint8Array.of(b));
  for (const [l, r] of merges) {
    const out = new Uint8Array(bytes[l].length + bytes[r].length);
    out.set(bytes[l]);
    out.set(bytes[r], bytes[l].length);
    bytes.push(out);
  }
  const tokens = bytes.map((b, i) => (i < N_SPECIAL ? SPECIAL_NAMES[i] : displayToken(b)));
  return { bytes, tokens };
}

/**
 * Train BPE on pre-token counts. Incremental pair statistics: each merge only revisits the words
 * that contain the merged pair. Ties break on the smaller (left, right) pair, so it is deterministic.
 */
export function trainBpe(
  wordCounts: Map<string, number>,
  vocabSize: number,
  log: (msg: string) => void = () => {},
): TokenizerData {
  const nMerges = vocabSize - MERGE_BASE;
  if (nMerges < 0) throw new Error(`vocabSize must be >= ${MERGE_BASE}`);
  const words: { syms: number[]; n: number }[] = [];
  for (const [w, n] of wordCounts) words.push({ syms: Array.from(enc.encode(w), (b) => b + BYTE_BASE), n });

  const K = 1 << 16;
  const pairCount = new Map<number, number>();
  const pairWords = new Map<number, Set<number>>();
  const addPairs = (wi: number, sign: 1 | -1) => {
    const { syms, n } = words[wi];
    for (let i = 0; i + 1 < syms.length; i++) {
      const k = syms[i] * K + syms[i + 1];
      pairCount.set(k, (pairCount.get(k) ?? 0) + sign * n);
      if (sign === 1) {
        let s = pairWords.get(k);
        if (!s) pairWords.set(k, (s = new Set()));
        s.add(wi);
      }
    }
  };
  for (let wi = 0; wi < words.length; wi++) addPairs(wi, 1);

  const merges: [number, number][] = [];
  for (let m = 0; m < nMerges; m++) {
    let best = -1;
    let bestN = 1; // need a count of at least 2
    for (const [k, n] of pairCount) {
      if (n > bestN || (n === bestN && best >= 0 && k < best)) {
        best = k;
        bestN = n;
      }
    }
    if (best < 0) {
      log(`stopped early: no pair occurs twice (${m} merges)`);
      break;
    }
    const a = Math.floor(best / K);
    const b = best % K;
    const id = MERGE_BASE + m;
    merges.push([a, b]);
    for (const wi of [...(pairWords.get(best) ?? [])]) {
      addPairs(wi, -1);
      const w = words[wi];
      const out: number[] = [];
      for (let i = 0; i < w.syms.length; i++) {
        if (i + 1 < w.syms.length && w.syms[i] === a && w.syms[i + 1] === b) {
          out.push(id);
          i++;
        } else out.push(w.syms[i]);
      }
      w.syms = out;
      addPairs(wi, 1);
    }
    pairCount.delete(best);
    pairWords.delete(best);
    for (const [k, n] of pairCount) if (n <= 0) pairCount.delete(k);
    if ((m + 1) % 250 === 0) log(`merge ${m + 1}/${nMerges}  last pair count ${bestN}`);
  }
  const { tokens } = buildTokens(merges);
  return {
    version: 1,
    type: "byte-level-bpe",
    vocabSize: MERGE_BASE + merges.length,
    specials: { bof: BOF, eof: EOF, pad: PAD },
    pretoken: { source: PRETOKEN_SOURCE, flags: PRETOKEN_FLAGS },
    merges,
    tokens,
  };
}

export class Tokenizer {
  readonly vocabSize: number;
  private readonly re: RegExp;
  private readonly rank = new Map<number, number>(); // pair key -> merge index
  private readonly bytes: Uint8Array[];
  readonly tokens: string[];
  private readonly cache = new Map<string, number[]>();

  constructor(readonly data: TokenizerData) {
    this.vocabSize = data.vocabSize;
    this.re = makePretokenRegex(data.pretoken.source, data.pretoken.flags);
    data.merges.forEach(([a, b], i) => this.rank.set(a * 65536 + b, i));
    const built = buildTokens(data.merges);
    this.bytes = built.bytes;
    this.tokens = built.tokens;
  }

  static fromJSON(json: string): Tokenizer {
    return new Tokenizer(JSON.parse(json) as TokenizerData);
  }

  toJSON(): string {
    return JSON.stringify(this.data);
  }

  pretokenize(text: string): string[] {
    this.re.lastIndex = 0;
    return text.match(this.re) ?? [];
  }

  private encodePiece(piece: string): number[] {
    const hit = this.cache.get(piece);
    if (hit) return hit;
    let syms = Array.from(enc.encode(piece), (b) => b + BYTE_BASE);
    while (syms.length > 1) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i + 1 < syms.length; i++) {
        const r = this.rank.get(syms[i] * 65536 + syms[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;
      const a = syms[bestAt];
      const b = syms[bestAt + 1];
      const id = MERGE_BASE + bestRank;
      const out: number[] = [];
      for (let i = 0; i < syms.length; i++) {
        if (i + 1 < syms.length && syms[i] === a && syms[i + 1] === b) {
          out.push(id);
          i++;
        } else out.push(syms[i]);
      }
      syms = out;
    }
    if (this.cache.size < 500_000) this.cache.set(piece, syms);
    return syms;
  }

  /** Encode text. Specials are never parsed out of the text; add them with the options. */
  encode(text: string, opts: { bof?: boolean; eof?: boolean } = {}): number[] {
    const out: number[] = [];
    if (opts.bof) out.push(BOF);
    for (const piece of this.pretokenize(text)) for (const id of this.encodePiece(piece)) out.push(id);
    if (opts.eof) out.push(EOF);
    return out;
  }

  /** Encode with the pre-token boundaries kept, for the lab's token chips. */
  encodePieces(text: string): { piece: string; ids: number[] }[] {
    return this.pretokenize(text).map((piece) => ({ piece, ids: this.encodePiece(piece) }));
  }

  tokenBytes(id: number): Uint8Array {
    return this.bytes[id];
  }

  /** Decode ids to text, skipping specials. Exact inverse of encode for any string. */
  decode(ids: ArrayLike<number>): string {
    let n = 0;
    for (let i = 0; i < ids.length; i++) n += this.bytes[ids[i]].length;
    const buf = new Uint8Array(n);
    let o = 0;
    for (let i = 0; i < ids.length; i++) {
      const b = this.bytes[ids[i]];
      buf.set(b, o);
      o += b.length;
    }
    return dec.decode(buf);
  }
}
