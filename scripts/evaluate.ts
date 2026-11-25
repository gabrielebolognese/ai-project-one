/**
 * Held-out evaluation of trained checkpoints against classic baselines.
 *
 *   npm run evaluate                                   # models/release vs baselines
 *   npm run evaluate -- --models models/release,models/lab-preview --windows 2700 --lines 500
 *
 * Everything is measured on the TEST split only (repos jsoup, gson, retrofit; never seen by the
 * tokenizer or the trainer). Baselines are built from the same TRAIN split the network saw.
 *
 * 1. Next-token prediction. `--windows` windows of ctx tokens are spread evenly over the test
 *    stream (files concatenated with <|bof|>/<|eof|>, exactly as data/test.bin). In each window the
 *    positions with at least `--min-ctx` tokens of context are scored, so every method predicts the
 *    same target tokens: top-1 and top-5 accuracy, mean NLL and perplexity, per repo and per
 *    token class.
 * 2. Rest-of-line completion. `--lines` non-blank lines are sampled from the test files (seeded).
 *    Each is cut at a random token boundary after its first content token; the method sees the
 *    file up to the cut (the model: last ctx-32 tokens, KV-cached greedy decoding) and must produce
 *    the rest of the line. Generation stops at the first newline, <|eof|> or 32 tokens. Exact match
 *    is on right-trimmed strings; edit similarity is 1 - Levenshtein / max length (CodeXGLUE).
 *    ms per completion is wall clock on one thread of this laptop, prompt prefill included.
 *
 * Baselines:
 *   - n-gram: interpolated modified Kneser-Ney token n-gram (order `--order`, default 5).
 *   - most-frequent: always predicts the most frequent training tokens (unigram ranking).
 *
 * Writes results/results.json and results/results.md (tables; hand-written findings are read from
 * results/findings.md if it exists).
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { cpus, totalmem } from "node:os";
import { parseArgs } from "node:util";
import { noGrad, Rng, kernelBackend } from "../src/engine/index.js";
import { loadModel, type Transformer } from "../src/model/index.js";
import { Tokenizer, BOF, EOF } from "../src/tokenizer/index.js";
import { IncrementalDecoder } from "../src/server/decoder.js";
import { buildNgram, cacheCounts, mixCache, ngramDist, ngramSizeBytes, type NgramModel } from "../src/eval/ngram.js";
import { loadTokenShard } from "../src/train/data.js";

const { values: a } = parseArgs({
  options: {
    models: { type: "string", default: "models/release" },
    windows: { type: "string", default: "2700" },
    "min-ctx": { type: "string", default: "32" },
    lines: { type: "string", default: "500" },
    order: { type: "string", default: "5" },
    seed: { type: "string", default: "2026" },
    data: { type: "string", default: "data" },
    out: { type: "string", default: "results" },
    "max-new": { type: "string", default: "32" },
  },
});

const t00 = performance.now();
const log = (m: string): void => console.log(`[eval ${((performance.now() - t00) / 1000).toFixed(0).padStart(4)}s] ${m}`);
const dataDir = a.data!;
const nWindows = Number(a.windows);
const minCtx = Number(a["min-ctx"]);
const nLines = Number(a.lines);
const order = Number(a.order);
const maxNew = Number(a["max-new"]);

// ------------------------------------------------------------------------------------------------
// Test corpus: per-file encodings with repo labels; must reproduce data/test.bin exactly.
// ------------------------------------------------------------------------------------------------
const tok = Tokenizer.fromJSON(readFileSync("models/tokenizer.json", "utf8"));
const V = tok.vocabSize;
interface TestFile { repo: string; path: string; text: string; ids: number[]; start: number }
const files: TestFile[] = [];
const streamArr: number[] = [];
for (const line of readFileSync(join(dataDir, "corpus", "test.jsonl"), "utf8").split("\n")) {
  if (!line.trim()) continue;
  const r = JSON.parse(line) as { repo: string; path: string; text: string };
  const ids = tok.encode(r.text);
  files.push({ repo: r.repo, path: r.path, text: r.text, ids, start: streamArr.length });
  streamArr.push(BOF, ...ids, EOF);
}
const stream = Uint16Array.from(streamArr);
const testBin = loadTokenShard(join(dataDir, "test.bin"));
let binMatches = testBin.length === stream.length;
if (binMatches) for (let i = 0; i < stream.length; i++) if (stream[i] !== testBin[i]) { binMatches = false; break; }
if (!binMatches) throw new Error("re-encoded test.jsonl does not match data/test.bin; refusing to evaluate on a different stream");
const repos = [...new Set(files.map((f) => f.repo))];
log(`test: ${files.length} files, ${repos.join(", ")}, ${stream.length.toLocaleString()} tokens (matches data/test.bin)`);

const fileStarts = files.map((f) => f.start);
function fileAt(pos: number): number {
  let lo = 0, hi = fileStarts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >>> 1;
    if (fileStarts[mid] <= pos) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

// Token classes for the breakdown.
const tokenText = (id: number): string => (id === BOF ? "<|bof|>" : id === EOF ? "<|eof|>" : id === 2 ? "<|pad|>" : tok.decode([id]));
type Cls = "identifier/keyword" | "whitespace" | "punctuation/operator" | "literal/other";
const CLASSES: Cls[] = ["identifier/keyword", "punctuation/operator", "whitespace", "literal/other"];
const clsOf: Cls[] = Array.from({ length: V }, (_, id) => {
  const s = tokenText(id);
  if (/^\s+$/.test(s)) return "whitespace";
  if (/^\s*[A-Za-z_$][A-Za-z0-9_$]*$/.test(s)) return "identifier/keyword";
  if (/^\s*[(){}\[\];,.<>=+\-*/%!&|^~?:@]+\s*$/.test(s)) return "punctuation/operator";
  return "literal/other";
});

// ------------------------------------------------------------------------------------------------
// Baselines from the TRAIN split.
// ------------------------------------------------------------------------------------------------
const train = loadTokenShard(join(dataDir, "train.bin"));
const uniCounts = new Float64Array(V);
for (let i = 0; i < train.length; i++) uniCounts[train[i]]++;
const uniLogP = new Float64Array(V);
for (let v = 0; v < V; v++) uniLogP[v] = Math.log((uniCounts[v] + 1) / (train.length + V)); // add-one
const uniOrder = Array.from({ length: V }, (_, v) => v).sort((x, y) => uniCounts[y] - uniCounts[x] || x - y);
log(`train: ${train.length.toLocaleString()} tokens; most frequent: ${uniOrder.slice(0, 5).map((v) => JSON.stringify(tokenText(v))).join(" ")}`);

const tb = performance.now();
const ngram: NgramModel = buildNgram(train, V, order, log);
const ngramBuildSec = (performance.now() - tb) / 1000;
log(`n-gram order ${order} built in ${ngramBuildSec.toFixed(1)} s, ${(ngramSizeBytes(ngram) / 1e6).toFixed(0)} MB`);

// Cache n-gram variant: the KN distribution mixed with a file-local cache (longest earlier match of
// up to CACHE_K context tokens within the same file, at most CACHE_BACK tokens back). The mixing
// strength beta is tuned on the VALIDATION split (commons-codec), never on test.
const CACHE_K = 3, CACHE_BACK = 2048;
const cacheC = new Float64Array(V);
const cacheTouched: number[] = [];
function fileStartBefore(s: ArrayLike<number>, pos: number): number {
  const lim = Math.max(0, pos - CACHE_BACK);
  for (let j = pos - 1; j >= lim; j--) if (s[j] === BOF) return j;
  return lim;
}
function cacheMixed(s: ArrayLike<number>, pos: number, base: Float64Array, out: Float64Array, beta: number, minFrom = 0): void {
  out.set(base);
  const total = cacheCounts(s, Math.max(minFrom, fileStartBefore(s, pos)), pos, CACHE_K, cacheC, cacheTouched);
  mixCache(out, cacheC, cacheTouched, total, beta);
}
const BETAS = [0.5, 1, 2, 4, 8, 16];
const cacheBeta = (() => {
  const val = loadTokenShard(join(dataDir, "val.bin"));
  const nW = 300, T0 = 128, sp = val.length - T0 - 1;
  const nll = new Float64Array(BETAS.length);
  const nllW = new Float64Array(BETAS.length);
  const base = new Float64Array(V), mixed = new Float64Array(V);
  for (let i = 0; i < nW; i++) {
    const o = Math.floor((i * sp) / (nW - 1));
    for (let t = minCtx; t < T0; t++) {
      const pos = o + t + 1;
      ngramDist(ngram, val, pos, base);
      const fs = fileStartBefore(val, pos);
      let total = cacheCounts(val, fs, pos, CACHE_K, cacheC, cacheTouched);
      BETAS.forEach((b, bi) => {
        mixed.set(base);
        mixCache(mixed, cacheC, cacheTouched, total, b);
        nll[bi] -= Math.log(mixed[val[pos]]);
      });
      total = cacheCounts(val, Math.max(fs, o), pos, CACHE_K, cacheC, cacheTouched);
      BETAS.forEach((b, bi) => {
        mixed.set(base);
        mixCache(mixed, cacheC, cacheTouched, total, b);
        nllW[bi] -= Math.log(mixed[val[pos]]);
      });
    }
  }
  const best = (x: Float64Array): number => { let bi = 0; for (let i = 1; i < x.length; i++) if (x[i] < x[bi]) bi = i; return bi; };
  const bi = best(nll), bw = best(nllW);
  const n = nW * (T0 - minCtx);
  log(`cache beta tuned on val (${n.toLocaleString()} tokens): file ${BETAS.map((b, i) => `${b}:${(nll[i] / n).toFixed(4)}`).join(" ")} -> ${BETAS[bi]}; window ${BETAS.map((b, i) => `${b}:${(nllW[i] / n).toFixed(4)}`).join(" ")} -> ${BETAS[bw]}`);
  return {
    beta: BETAS[bi], betaWindow: BETAS[bw],
    valNll: Object.fromEntries(BETAS.map((b, i) => [String(b), nll[i] / n])),
    valNllWindow: Object.fromEntries(BETAS.map((b, i) => [String(b), nllW[i] / n])),
  };
})();

// ------------------------------------------------------------------------------------------------
// Metric accumulators.
// ------------------------------------------------------------------------------------------------
interface Acc { n: number; top1: number; top5: number; nll: number }
const newAcc = (): Acc => ({ n: 0, top1: 0, top5: 0, nll: 0 });
class NextTokenStats {
  all = newAcc();
  perRepo = new Map<string, Acc>(repos.map((r) => [r, newAcc()]));
  perClass = new Map<Cls, Acc>(CLASSES.map((c) => [c, newAcc()]));
  add(rank: number, nll: number, repo: string, cls: Cls): void {
    for (const acc of [this.all, this.perRepo.get(repo)!, this.perClass.get(cls)!]) {
      acc.n++;
      if (rank === 0) acc.top1++;
      if (rank < 5) acc.top5++;
      acc.nll += nll;
    }
  }
  static fmt(acc: Acc) {
    const nll = acc.nll / Math.max(1, acc.n);
    return { n: acc.n, top1: acc.top1 / Math.max(1, acc.n), top5: acc.top5 / Math.max(1, acc.n), nll, ppl: Math.exp(nll) };
  }
  toJSON() {
    return {
      ...NextTokenStats.fmt(this.all),
      perRepo: Object.fromEntries([...this.perRepo].map(([k, v]) => [k, NextTokenStats.fmt(v)])),
      perClass: Object.fromEntries([...this.perClass].map(([k, v]) => [k, NextTokenStats.fmt(v)])),
    };
  }
}

/** Rank of `target` under argmax-first-index tie breaking. */
function rankOf(scores: ArrayLike<number>, off: number, target: number): number {
  const st = scores[off + target];
  let r = 0;
  for (let v = 0; v < V; v++) {
    const s = scores[off + v];
    if (s > st || (s === st && v < target)) r++;
  }
  return r;
}

// Scoring windows (shared by every method).
const models = a.models!.split(",").map((s) => s.trim()).filter(Boolean);
const loaded = models.map((dir) => {
  if (!existsSync(join(dir, "weights.bin"))) throw new Error(`${dir}/weights.bin not found`);
  const { model, config } = loadModel(dir);
  if (model.cfg.vocabSize !== V) throw new Error(`${dir}: vocab ${model.cfg.vocabSize} != tokenizer ${V}`);
  // Hash the exact bytes that are evaluated, so every number is tied to one weights file.
  const weightsSha256 = createHash("sha256").update(readFileSync(join(dir, "weights.bin"))).digest("hex");
  return { dir, name: dir.replace(/\\/g, "/").split("/").pop()!, model, config, weightsSha256 };
});
for (const l of loaded) log(`[${l.name}] weights.bin sha256 ${l.weightsSha256}`);
const T = Math.min(...loaded.map((l) => l.model.cfg.ctx), 128);

// Machine-wide CPU load from OTHER processes while a timed phase runs (os.cpus() totals minus this
// process's own CPU time). Reported next to the latency numbers so contention is visible.
function cpuSnap(): { busy: number; total: number; own: number } {
  let busy = 0, total = 0;
  for (const c of cpus()) { const t = c.times; busy += t.user + t.nice + t.sys + t.irq; total += t.user + t.nice + t.sys + t.irq + t.idle; }
  const u = process.cpuUsage();
  return { busy, total, own: (u.user + u.system) / 1000 };
}
function otherLoad(a0: ReturnType<typeof cpuSnap>, a1: ReturnType<typeof cpuSnap>): { machineBusyPct: number; otherProcessesCores: number } {
  const total = a1.total - a0.total, busy = a1.busy - a0.busy, own = a1.own - a0.own;
  const wall = total / cpus().length;
  return { machineBusyPct: +((100 * busy) / total).toFixed(1), otherProcessesCores: +(Math.max(0, busy - own) / wall).toFixed(2) };
}
if (minCtx >= T) throw new Error("--min-ctx must be below ctx");
const span = stream.length - T - 1;
const offsets = Array.from({ length: nWindows }, (_, i) => Math.floor((i * span) / Math.max(1, nWindows - 1)));
const scoredPerWindow = T - minCtx;
log(`next-token: ${nWindows} windows x ${scoredPerWindow} scored positions = ${(nWindows * scoredPerWindow).toLocaleString()} target tokens (T=${T}, min ctx ${minCtx})`);

function evalModelNextToken(model: Transformer): { stats: NextTokenStats; msPerWindow: number } {
  const stats = new NextTokenStats();
  const B = 8;
  const t0 = performance.now();
  noGrad(() => {
    for (let w = 0; w < offsets.length; w += B) {
      const part = offsets.slice(w, w + B);
      const ids = new Int32Array(part.length * T);
      part.forEach((o, b) => { for (let t = 0; t < T; t++) ids[b * T + t] = stream[o + t]; });
      const { logits } = model.forward(ids, part.length, T);
      const L = logits.data;
      part.forEach((o, b) => {
        for (let t = minCtx; t < T; t++) {
          const row = (b * T + t) * V;
          const target = stream[o + t + 1];
          let mx = -Infinity;
          for (let v = 0; v < V; v++) if (L[row + v] > mx) mx = L[row + v];
          let z = 0;
          for (let v = 0; v < V; v++) z += Math.exp(L[row + v] - mx);
          const nll = Math.log(z) + mx - L[row + target];
          stats.add(rankOf(L, row, target), nll, files[fileAt(o + t + 1)].repo, clsOf[target]);
        }
      });
    }
  });
  return { stats, msPerWindow: (performance.now() - t0) / offsets.length };
}

function evalNgramNextToken(): { ng: NextTokenStats; ngc: NextTokenStats; ngw: NextTokenStats; uni: NextTokenStats; msPerToken: number; msPerTokenCache: number } {
  const ng = new NextTokenStats();
  const ngc = new NextTokenStats();
  const ngw = new NextTokenStats();
  const pc = new Float64Array(V);
  let tC = 0;
  const uni = new NextTokenStats();
  const uniRank = new Int32Array(V);
  uniOrder.forEach((v, r) => { uniRank[v] = r; });
  const p = new Float64Array(V);
  let tNg = 0, nNg = 0;
  for (const o of offsets) {
    for (let t = minCtx; t < T; t++) {
      const pos = o + t + 1;
      const target = stream[pos];
      const repo = files[fileAt(pos)].repo;
      const ts = performance.now();
      ngramDist(ngram, stream, pos, p);
      tNg += performance.now() - ts;
      nNg++;
      ng.add(rankOf(p, 0, target), -Math.log(p[target]), repo, clsOf[target]);
      const tc = performance.now();
      cacheMixed(stream, pos, p, pc, cacheBeta.beta);
      tC += performance.now() - tc;
      ngc.add(rankOf(pc, 0, target), -Math.log(pc[target]), repo, clsOf[target]);
      cacheMixed(stream, pos, p, pc, cacheBeta.betaWindow, o);
      ngw.add(rankOf(pc, 0, target), -Math.log(pc[target]), repo, clsOf[target]);
      uni.add(uniRank[target], -uniLogP[target], repo, clsOf[target]);
    }
  }
  return { ng, ngc, ngw, uni, msPerToken: tNg / nNg, msPerTokenCache: (tNg + tC) / nNg };
}

// ------------------------------------------------------------------------------------------------
// Line-completion samples.
// ------------------------------------------------------------------------------------------------
interface LineSample { file: number; line: number; cut: number; prefixIds: number[]; target: string; kind: "code" | "comment"; lineText: string }
function sampleLines(): LineSample[] {
  const rng = new Rng(Number(a.seed));
  const cands: { file: number; lineStart: number; lineEnd: number; line: number }[] = [];
  files.forEach((f, fi) => {
    // eslint-disable-next-line no-control-regex
    if (/[^\x00-\x7f]/.test(f.text)) return; // byte offsets == char offsets only for ASCII files
    let s = 0, ln = 0;
    while (s <= f.text.length) {
      let e = f.text.indexOf("\n", s);
      if (e < 0) e = f.text.length;
      if (f.text.slice(s, e).trim().length >= 2) cands.push({ file: fi, lineStart: s, lineEnd: e, line: ln });
      s = e + 1;
      ln++;
    }
  });
  // Shuffle candidates (Fisher-Yates) and take the first nLines that admit a valid cut.
  for (let i = cands.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [cands[i], cands[j]] = [cands[j], cands[i]];
  }
  const tokStarts = new Map<number, Int32Array>();
  const startsOf = (fi: number): Int32Array => {
    let st = tokStarts.get(fi);
    if (!st) {
      const ids = files[fi].ids;
      st = new Int32Array(ids.length + 1);
      for (let k = 0; k < ids.length; k++) st[k + 1] = st[k] + tok.tokenBytes(ids[k]).length;
      tokStarts.set(fi, st);
    }
    return st;
  };
  const out: LineSample[] = [];
  for (const c of cands) {
    if (out.length >= nLines) break;
    const f = files[c.file];
    const st = startsOf(c.file);
    const lineText = f.text.slice(c.lineStart, c.lineEnd);
    const contentStart = c.lineStart + (lineText.length - lineText.trimStart().length);
    const valid: number[] = [];
    for (let k = 1; k < f.ids.length; k++) {
      const s = st[k];
      if (s <= contentStart) continue;
      if (s >= c.lineEnd) break;
      if (f.text.slice(s, c.lineEnd).trim().length > 0) valid.push(k);
    }
    if (!valid.length) continue;
    const k = valid[rng.int(valid.length)];
    const trimmed = lineText.trim();
    out.push({
      file: c.file, line: c.line, cut: k, prefixIds: [BOF, ...f.ids.slice(0, k)],
      target: f.text.slice(st[k], c.lineEnd), lineText,
      kind: /^(\/\/|\/\*|\*)/.test(trimmed) ? "comment" : "code",
    });
  }
  return out;
}

function lev(x: string, y: string): number {
  if (x === y) return 0;
  const m = x.length, n = y.length;
  let prev = Array.from({ length: n + 1 }, (_, j) => j);
  for (let i = 1; i <= m; i++) {
    const cur = [i];
    for (let j = 1; j <= n; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x[i - 1] === y[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[n];
}
const editSim = (p: string, t: string): number => {
  const x = p.trimEnd(), y = t.trimEnd();
  const L = Math.max(x.length, y.length);
  return L === 0 ? 1 : 1 - lev(x, y) / L;
};

/** Accumulate generated tokens into a line; returns true when generation must stop. */
function pushToken(state: { text: string; done: boolean }, id: number): void {
  if (id === EOF || id === BOF || id === 2) { state.done = true; return; }
  state.text += tokenText(id);
  if (state.text.includes("\n")) { state.text = state.text.slice(0, state.text.indexOf("\n")); state.done = true; }
}

const argmax = (x: ArrayLike<number>): number => {
  let b = 0;
  for (let v = 1; v < x.length; v++) if (x[v] > x[b]) b = v;
  return b;
};

function completeModel(dec: IncrementalDecoder, prefix: number[]): { text: string; steps: number } {
  const P = dec.ctx - maxNew;
  let logits = dec.prefill(prefix.slice(-P));
  const s = { text: "", done: false };
  let steps = 0;
  for (let i = 0; i < maxNew; i++) {
    const id = argmax(logits);
    pushToken(s, id);
    steps++;
    if (s.done || i === maxNew - 1) break;
    logits = dec.step(id);
  }
  return { text: s.text, steps };
}

function completeNgram(prefix: number[]): string {
  const hist = prefix.slice(-(order - 1));
  const p = new Float64Array(V);
  const s = { text: "", done: false };
  for (let i = 0; i < maxNew && !s.done; i++) {
    ngramDist(ngram, hist, hist.length, p);
    const id = argmax(p);
    pushToken(s, id);
    hist.push(id);
  }
  return s.text;
}

function completeNgramCache(prefix: number[], window = false): string {
  const hist = prefix.slice(window ? -(T - maxNew) : -CACHE_BACK);
  const p = new Float64Array(V), pc = new Float64Array(V);
  const s = { text: "", done: false };
  for (let i = 0; i < maxNew && !s.done; i++) {
    ngramDist(ngram, hist, hist.length, p);
    // window mode: hist holds only the network's prompt (last ctx-maxNew tokens) plus what was generated
    cacheMixed(hist, hist.length, p, pc, window ? cacheBeta.betaWindow : cacheBeta.beta);
    const id = argmax(pc);
    pushToken(s, id);
    hist.push(id);
  }
  return s.text;
}

function completeMostFrequent(): string {
  const s = { text: "", done: false };
  for (let i = 0; i < maxNew && !s.done; i++) pushToken(s, uniOrder[0]);
  return s.text;
}

interface LineResult { em: number; editSim: number; n: number; ms: number[]; byKind: Record<string, { n: number; em: number; editSim: number }>; preds: string[] }
function scoreLines(samples: LineSample[], run: (s: LineSample) => string): LineResult {
  const res: LineResult = { em: 0, editSim: 0, n: samples.length, ms: [], byKind: {}, preds: [] };
  for (const s of samples) {
    const t0 = performance.now();
    const pred = run(s);
    res.ms.push(performance.now() - t0);
    const em = pred.trimEnd() === s.target.trimEnd() ? 1 : 0;
    const es = editSim(pred, s.target);
    res.em += em;
    res.editSim += es;
    const k = (res.byKind[s.kind] ??= { n: 0, em: 0, editSim: 0 });
    k.n++; k.em += em; k.editSim += es;
    res.preds.push(pred);
  }
  return res;
}
const pct = (xs: number[], q: number): number => {
  const s = [...xs].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
};
function lineSummary(r: LineResult) {
  return {
    n: r.n,
    exactMatch: r.em / r.n,
    editSim: r.editSim / r.n,
    byKind: Object.fromEntries(Object.entries(r.byKind).map(([k, v]) => [k, { n: v.n, exactMatch: v.em / v.n, editSim: v.editSim / v.n }])),
    msPerCompletion: { mean: r.ms.reduce((x, y) => x + y, 0) / r.n, p50: pct(r.ms, 0.5), p95: pct(r.ms, 0.95) },
  };
}

// ------------------------------------------------------------------------------------------------
// Run.
// ------------------------------------------------------------------------------------------------
const samples = sampleLines();
const kinds = samples.reduce<Record<string, number>>((m, s) => ((m[s.kind] = (m[s.kind] ?? 0) + 1), m), {});
log(`line completion: ${samples.length} sampled lines (${JSON.stringify(kinds)}), mean target ${(samples.reduce((x, s) => x + s.target.trimEnd().length, 0) / samples.length).toFixed(1)} chars`);

type MethodResult = {
  name: string;
  kind: "network" | "baseline";
  description: string;
  params?: number;
  sizeMB: number;
  weightsSha256?: string;
  training?: Record<string, unknown>;
  nextToken: ReturnType<NextTokenStats["toJSON"]> & { msPerToken: number };
  line: ReturnType<typeof lineSummary>;
};
const methods: MethodResult[] = [];
const allPreds: Record<string, string[]> = {};

for (const l of loaded) {
  log(`[${l.name}] next-token on ${nWindows} windows...`);
  const { stats, msPerWindow } = evalModelNextToken(l.model);
  const nt = stats.toJSON();
  log(`[${l.name}] top1 ${(nt.top1 * 100).toFixed(2)}% top5 ${(nt.top5 * 100).toFixed(2)}% ppl ${nt.ppl.toFixed(3)} (${msPerWindow.toFixed(1)} ms/window)`);
  const dec = new IncrementalDecoder(l.model);
  completeModel(dec, samples[0].prefixIds); // warm-up (JIT, WASM instantiation)
  let steps = 0;
  const c0 = cpuSnap();
  const lr = scoreLines(samples, (s) => { const r = completeModel(dec, s.prefixIds); steps += r.steps; return r.text; });
  const ls = lineSummary(lr);
  const load = otherLoad(c0, cpuSnap());
  log(`[${l.name}] background load during line completion: machine ${load.machineBusyPct}% busy, other processes ${load.otherProcessesCores} cores`);
  log(`[${l.name}] line EM ${(ls.exactMatch * 100).toFixed(1)}% editSim ${ls.editSim.toFixed(3)} ${ls.msPerCompletion.mean.toFixed(1)} ms/completion (${(steps / samples.length).toFixed(1)} tokens avg)`);
  const cfgAny = l.config as Record<string, unknown>;
  const summary = (cfgAny.summary ?? {}) as Record<string, number>;
  const trainCfg = (cfgAny.train ?? {}) as Record<string, unknown>;
  methods.push({
    name: `transformer (${l.name})`,
    kind: "network",
    description: `${l.model.cfg.nLayers}L d${l.model.cfg.dModel} ${l.model.cfg.nHeads}H ctx${l.model.cfg.ctx}, tied embeddings, trained from scratch in TypeScript on this CPU`,
    params: l.model.numParams,
    sizeMB: (l.model.numParams * 4) / 1e6,
    weightsSha256: l.weightsSha256,
    training: {
      dir: l.dir.replace(/\\/g, "/"),
      status: cfgAny.status,
      checkpoint: cfgAny.best,
      note: (cfgAny.summary as Record<string, unknown> | undefined)?.stopNote,
      budgetMin: trainCfg.budgetMin,
      elapsedMin: summary.elapsedSec !== undefined ? summary.elapsedSec / 60 : undefined,
      tokensSeen: summary.tokens,
      steps: summary.steps,
      tokensPerSec: summary.tokensPerSec,
      tokensPerParam: summary.tokens !== undefined ? summary.tokens / l.model.numParams : undefined,
      bestValLoss: summary.bestValLoss,
      bestStep: summary.bestStep,
      peakLr: (trainCfg.lr as Record<string, number> | undefined)?.peakLr,
      workers: trainCfg.workers,
      batchPerWorker: trainCfg.batchPerWorker,
      config: l.model.cfg,
    },
    nextToken: { ...nt, msPerToken: msPerWindow / T },
    line: { ...ls, avgGeneratedTokens: steps / samples.length, backgroundLoad: load } as ReturnType<typeof lineSummary>,
  });
  allPreds[methods.at(-1)!.name] = lr.preds;
}

log(`n-gram + most-frequent next-token...`);
const { ng, ngc, ngw, uni, msPerToken: ngMs, msPerTokenCache: ngcMs } = evalNgramNextToken();
const ngLine = scoreLines(samples, (s) => completeNgram(s.prefixIds));
const ngcLine = scoreLines(samples, (s) => completeNgramCache(s.prefixIds));
const ngwLine = scoreLines(samples, (s) => completeNgramCache(s.prefixIds, true));
const uniLine = scoreLines(samples, () => completeMostFrequent());
const ngName = `${order}-gram Kneser-Ney`;
methods.push({
  name: ngName,
  kind: "baseline",
  description: `interpolated modified Kneser-Ney token ${order}-gram on the same ${(train.length / 1e6).toFixed(1)}M train tokens`,
  sizeMB: ngramSizeBytes(ngram) / 1e6,
  nextToken: { ...ng.toJSON(), msPerToken: ngMs },
  line: lineSummary(ngLine),
});
allPreds[ngName] = ngLine.preds;
const ngcName = `${order}-gram Kneser-Ney + file cache`;
methods.push({
  name: ngcName,
  kind: "baseline",
  description: `the ${order}-gram mixed with a file-local cache (longest earlier match of up to ${CACHE_K} tokens, up to ${CACHE_BACK} tokens back in the same file; beta=${cacheBeta.beta} tuned on val)`,
  sizeMB: ngramSizeBytes(ngram) / 1e6,
  nextToken: { ...ngc.toJSON(), msPerToken: ngcMs },
  line: lineSummary(ngcLine),
});
allPreds[ngcName] = ngcLine.preds;
const ngwName = `${order}-gram Kneser-Ney + window cache`;
methods.push({
  name: ngwName,
  kind: "baseline",
  description: `the ${order}-gram mixed with a cache limited to the same tokens the network sees (the window before the target, at most ${T} tokens; ${T - maxNew} for line completion; beta=${cacheBeta.betaWindow} tuned on val)`,
  sizeMB: ngramSizeBytes(ngram) / 1e6,
  nextToken: { ...ngw.toJSON(), msPerToken: ngcMs },
  line: lineSummary(ngwLine),
});
allPreds[ngwName] = ngwLine.preds;
methods.push({
  name: "most-frequent token",
  kind: "baseline",
  description: "always predicts the most frequent training tokens (unigram ranking; add-one probabilities for perplexity)",
  sizeMB: (V * 8) / 1e6,
  nextToken: { ...uni.toJSON(), msPerToken: 0 },
  line: lineSummary(uniLine),
});
allPreds["most-frequent token"] = uniLine.preds;
for (const m of methods.slice(-4)) {
  log(`[${m.name}] top1 ${(m.nextToken.top1 * 100).toFixed(2)}% top5 ${(m.nextToken.top5 * 100).toFixed(2)}% ppl ${m.nextToken.ppl.toFixed(3)} | line EM ${(m.line.exactMatch * 100).toFixed(1)}% editSim ${m.line.editSim.toFixed(3)}`);
}

// Paired comparison on lines: first network vs n-gram.
const net0 = methods.find((m) => m.kind === "network");
let paired: Record<string, number> | undefined;
if (net0) {
  const p0 = allPreds[net0.name], p1 = allPreds[ngName];
  let both = 0, onlyNet = 0, onlyNg = 0, neither = 0;
  samples.forEach((s, i) => {
    const e0 = p0[i].trimEnd() === s.target.trimEnd(), e1 = p1[i].trimEnd() === s.target.trimEnd();
    if (e0 && e1) both++; else if (e0) onlyNet++; else if (e1) onlyNg++; else neither++;
  });
  // exact two-sided sign test on the discordant pairs
  const nD = onlyNet + onlyNg, kMin = Math.min(onlyNet, onlyNg);
  let tail = 0;
  const logC = (n: number, k: number): number => { let s = 0; for (let i = 1; i <= k; i++) s += Math.log((n - k + i) / i); return s; };
  for (let k = 0; k <= kMin; k++) tail += Math.exp(logC(nD, k) - nD * Math.log(2));
  paired = { both, onlyNetwork: onlyNet, onlyNgram: onlyNg, neither, signTestP: Math.min(1, 2 * tail) };
}

const examples = samples.slice(0, 12).map((s, i) => ({
  file: `${files[s.file].repo}/${files[s.file].path}`,
  line: s.line + 1,
  kind: s.kind,
  prefix: s.lineText.slice(0, s.lineText.length - s.target.length),
  target: s.target,
  predictions: Object.fromEntries(Object.entries(allPreds).map(([k, v]) => [k, v[i]])),
}));

const results = {
  date: new Date().toISOString(),
  machine: { cpu: cpus()[0]?.model.trim(), logicalCores: cpus().length, totalMemGB: +(totalmem() / 2 ** 30).toFixed(1), node: process.version, kernels: kernelBackend(), evalThreads: 1 },
  protocol: {
    split: "test (held-out repos, never used for tokenizer or model training)",
    repos,
    files: files.length,
    testTokens: stream.length,
    nextToken: { windows: nWindows, windowTokens: T, minContext: minCtx, scoredTokens: nWindows * scoredPerWindow },
    lineCompletion: { lines: samples.length, kinds, seed: Number(a.seed), maxNewTokens: maxNew, modelPromptTokens: `last ctx-${maxNew}`, ngramContextTokens: order - 1, exactMatch: "right-trimmed string equality", editSim: "1 - Levenshtein/max(len)" },
    baselinesTrainedOn: { split: "train", tokens: train.length },
  },
  ngram: {
    order,
    smoothing: "interpolated modified Kneser-Ney (Chen & Goodman 1998), counts from data/train.bin",
    buildSec: ngramBuildSec,
    uniqueNgrams: ngram.tables.map((t) => t.counts.length),
    discounts: ngram.tables.map((t) => t.D.map((d) => +d.toFixed(4))),
    cache: { maxContext: CACHE_K, maxBack: CACHE_BACK, beta: cacheBeta.beta, betaWindow: cacheBeta.betaWindow, valNllByBeta: cacheBeta.valNll, valNllByBetaWindow: cacheBeta.valNllWindow },
  },
  methods,
  pairedLineComparison: paired,
  examples,
  elapsedSec: (performance.now() - t00) / 1000,
};
mkdirSync(a.out!, { recursive: true });
writeFileSync(join(a.out!, "results.json"), JSON.stringify(results, null, 2));

// ------------------------------------------------------------------------------------------------
// Markdown.
// ------------------------------------------------------------------------------------------------
const P = (x: number, d = 1): string => `${(x * 100).toFixed(d)}%`;
const md: string[] = [];
md.push("# Java Autocomplete Lab: evaluation", "");
const nets = methods.filter((m) => m.kind === "network");
const rel = nets[0];
const tr = rel?.training as Record<string, number | undefined> | undefined;
md.push(
  `Held-out test repos (${repos.join(", ")}): ${files.length} files, ${stream.length.toLocaleString()} tokens, never seen by the tokenizer or the trainer. ` +
    `Next-token metrics use ${nWindows.toLocaleString()} windows of ${T} tokens spread evenly over the test stream, scoring the ${scoredPerWindow} positions per window that have at least ${minCtx} tokens of context ` +
    `(${(nWindows * scoredPerWindow).toLocaleString()} target tokens, identical for every method). Line completion uses ${samples.length} seeded random lines cut at a random token boundary (greedy decoding, stop at newline).` +
    (rel && tr ? ` Model: ${(rel.params! / 1e6).toFixed(2)}M parameters, trained ${tr.elapsedMin?.toFixed(1)} min on this laptop's CPU (Ryzen 7 5700U, ${tr.workers} worker threads), ${((tr.tokensSeen ?? 0) / 1e6).toFixed(1)}M tokens seen (${tr.tokensPerParam?.toFixed(1)} tokens/param).` : ""),
  "",
);
md.push("## Main results", "");
md.push("| Method | Params / size | Top-1 | Top-5 | Perplexity | Line exact match | Line edit sim | ms/completion (mean / p95) |", "|---|---|---|---|---|---|---|---|");
for (const m of methods) {
  const size = m.params ? `${(m.params / 1e6).toFixed(2)}M (${m.sizeMB.toFixed(1)} MB)` : `${m.sizeMB.toFixed(0)} MB`;
  md.push(`| ${m.name} | ${size} | ${P(m.nextToken.top1)} | ${P(m.nextToken.top5)} | ${m.nextToken.ppl.toFixed(2)} | ${P(m.line.exactMatch)} | ${m.line.editSim.toFixed(3)} | ${m.line.msPerCompletion.mean.toFixed(1)} / ${m.line.msPerCompletion.p95.toFixed(1)} |`);
}
md.push("");
if (paired) {
  md.push(
    `Paired line comparison (${net0!.name} vs ${ngName}, same ${samples.length} lines): both exact ${paired.both}, only network ${paired.onlyNetwork}, only n-gram ${paired.onlyNgram}, neither ${paired.neither}; exact two-sided sign test on discordant lines p = ${paired.signTestP.toPrecision(2)}.`,
    "",
  );
}
md.push("## Top-1 by test repo", "");
md.push(`| Method | ${repos.join(" | ")} |`, `|---|${repos.map(() => "---").join("|")}|`);
for (const m of methods) md.push(`| ${m.name} | ${repos.map((r) => `${P(m.nextToken.perRepo[r].top1)} (ppl ${m.nextToken.perRepo[r].ppl.toFixed(2)})`).join(" | ")} |`);
md.push("");
md.push("## Top-1 by target token class", "");
const cnt0 = methods[0].nextToken.perClass;
md.push(`| Method | ${CLASSES.map((c) => `${c} (${P(cnt0[c].n / methods[0].nextToken.n, 0)} of tokens)`).join(" | ")} |`, `|---|${CLASSES.map(() => "---").join("|")}|`);
for (const m of methods) md.push(`| ${m.name} | ${CLASSES.map((c) => P(m.nextToken.perClass[c].top1)).join(" | ")} |`);
md.push("");
md.push("## Line completion by line kind", "");
const kindNames = Object.keys(kinds);
md.push(`| Method | ${kindNames.map((k) => `${k} (n=${kinds[k]}) EM / edit sim`).join(" | ")} |`, `|---|${kindNames.map(() => "---").join("|")}|`);
for (const m of methods) md.push(`| ${m.name} | ${kindNames.map((k) => `${P(m.line.byKind[k]?.exactMatch ?? 0)} / ${(m.line.byKind[k]?.editSim ?? 0).toFixed(3)}`).join(" | ")} |`);
md.push("");
if (nets.length) {
  md.push("## Training runs", "");
  md.push("| Run | Params | Budget | Wall clock | Tokens seen | Tokens/param | Train tok/s | Peak LR | Best val loss (commons-codec) |", "|---|---|---|---|---|---|---|---|---|");
  for (const m of nets) {
    const t = m.training as Record<string, number | undefined>;
    md.push(`| ${m.name} | ${(m.params! / 1e6).toFixed(2)}M | ${t.budgetMin} min | ${t.elapsedMin?.toFixed(1)} min | ${((t.tokensSeen ?? 0) / 1e6).toFixed(2)}M | ${t.tokensPerParam?.toFixed(1)} | ${t.tokensPerSec?.toFixed(0)} | ${t.peakLr} | ${t.bestValLoss?.toFixed(4)} (step ${t.bestStep}) |`);
  }
  md.push("");
  md.push("Evaluated weights (sha256 of weights.bin):", "");
  for (const m of nets) {
    const ld = (m.line as unknown as { backgroundLoad?: { machineBusyPct: number; otherProcessesCores: number } }).backgroundLoad;
    md.push(`- ${m.name}: \`${m.weightsSha256}\`` + (ld ? `; background load while timing line completion: machine ${ld.machineBusyPct}% busy, ${ld.otherProcessesCores} cores used by other processes` : ""));
  }
  md.push("");
  for (const m of nets) {
    const note = (m.training as Record<string, unknown>).note;
    if (typeof note === "string") md.push(`Note on ${m.name}: ${note}`, "");
  }
}
const pilotsPath = join(a.out!, "pilots.json");
if (existsSync(pilotsPath)) {
  const pj = JSON.parse(readFileSync(pilotsPath, "utf8")) as { note: string; runs: Record<string, number | string>[] };
  md.push("## LR / batch pilots (6 min each, before the release run)", "", pj.note, "");
  md.push("| Run | Peak LR | Batch/worker | Tokens/step (T=128) | Steps | Tokens | tok/s | Best val loss (EMA) |", "|---|---|---|---|---|---|---|---|");
  for (const r of pj.runs) md.push(`| ${r.run} | ${r.peakLr} | ${r.batchPerWorker} | ${r.tokensPerStepAtT128} | ${r.steps} | ${(Number(r.tokens) / 1e6).toFixed(2)}M | ${r.tokensPerSec} | ${Number(r.bestValLossEma).toFixed(3)} |`);
  md.push("");
}
md.push("## n-gram baseline", "");
md.push(`Order ${order}, interpolated modified Kneser-Ney, built in ${ngramBuildSec.toFixed(1)} s from ${train.length.toLocaleString()} train tokens; unique n-grams per order: ${ngram.tables.map((t) => t.counts.length.toLocaleString()).join(" / ")}. ${(ngramSizeBytes(ngram) / 1e6).toFixed(0)} MB in memory, against ${rel ? rel.sizeMB.toFixed(1) : "?"} MB for the network. ` +
  `The cache variant (Tu et al. 2014; Hellendoorn & Devanbu 2017) mixes in the tokens that followed the longest matching context (up to ${CACHE_K} tokens) earlier in the same file, up to ${CACHE_BACK} tokens back, so it sees more of the file than the network's ${T}-token window. The window-cache variant only looks at the tokens the network also sees (the same window), which makes it the like-for-like comparison. Mixing weight total/(total+beta), beta=${cacheBeta.beta} (file) and ${cacheBeta.betaWindow} (window), chosen on the validation repo from {${BETAS.join(", ")}}.`, "");
md.push("## Examples (first 12 sampled lines)", "");
md.push("| Prefix (line) | Target | " + Object.keys(allPreds).join(" | ") + " |", "|---|---|" + Object.keys(allPreds).map(() => "---").join("|") + "|");
const esc = (s: string): string => "`" + s.replace(/\|/g, "\\|").replace(/`/g, "'").trim() + "`";
for (const e of examples) md.push(`| ${esc(e.prefix)} | ${esc(e.target)} | ${Object.values(e.predictions).map((p) => esc(p) || "(empty)").join(" | ")} |`);
md.push("");
const findingsPath = join(a.out!, "findings.md");
if (existsSync(findingsPath)) md.push(readFileSync(findingsPath, "utf8").trim(), "");
md.push(`Reproduce: \`npm run evaluate -- --models ${models.join(",")} --windows ${nWindows} --lines ${nLines} --order ${order} --seed ${a.seed}\` (${(results.elapsedSec / 60).toFixed(1)} min, one thread). Raw numbers: results/results.json.`, "");
writeFileSync(join(a.out!, "results.md"), md.join("\n"));
log(`wrote ${join(a.out!, "results.json")} and results.md`);
