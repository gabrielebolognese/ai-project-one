import { test } from "node:test";
import assert from "node:assert/strict";
import { buildNgram, cacheCounts, mixCache, ngramDist } from "./ngram.js";
import { Rng } from "../engine/rng.js";

function randomStream(n: number, V: number, seed: number): Uint16Array {
  const rng = new Rng(seed);
  const s = new Uint16Array(n);
  let prev = 0;
  for (let i = 0; i < n; i++) {
    // mildly structured: next token often prev+1
    prev = rng.next() < 0.6 ? (prev + 1) % V : rng.int(V);
    s[i] = prev;
  }
  return s;
}

test("n-gram tables hold the right raw and continuation counts", () => {
  const s = randomStream(3000, 12, 3);
  const m = buildNgram(s, 12, 3);
  // brute-force raw trigram counts
  const raw = new Map<string, number>();
  for (let i = 0; i + 3 <= s.length; i++) {
    const k = `${s[i]},${s[i + 1]},${s[i + 2]}`;
    raw.set(k, (raw.get(k) ?? 0) + 1);
  }
  const t3 = m.tables[2];
  assert.equal(t3.counts.length, raw.size);
  for (let r = 0; r < t3.counts.length; r++) {
    const k = `${t3.grams[r * 3]},${t3.grams[r * 3 + 1]},${t3.grams[r * 3 + 2]}`;
    assert.equal(t3.counts[r], raw.get(k));
  }
  // continuation counts of bigrams = distinct left extensions among trigrams
  const cont = new Map<string, Set<number>>();
  for (const k of raw.keys()) {
    const [a, b, c] = k.split(",").map(Number);
    const key = `${b},${c}`;
    if (!cont.has(key)) cont.set(key, new Set());
    cont.get(key)!.add(a);
  }
  const t2 = m.tables[1];
  assert.equal(t2.counts.length, cont.size);
  for (let r = 0; r < t2.counts.length; r++) {
    assert.equal(t2.counts[r], cont.get(`${t2.grams[r * 2]},${t2.grams[r * 2 + 1]}`)!.size);
  }
});

test("n-gram distributions are normalised and favour seen continuations", () => {
  const V = 16;
  const s = randomStream(5000, V, 7);
  const m = buildNgram(s, V, 4);
  for (const hist of [[1, 2, 3], [0], [], [5, 5, 5], [15, 0, 1]]) {
    const p = ngramDist(m, hist, hist.length);
    const sum = p.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-9, `sum ${sum}`);
    for (const v of p) assert.ok(v > 0);
  }
  const p = ngramDist(m, [1, 2, 3], 3);
  let arg = 0;
  for (let v = 1; v < V; v++) if (p[v] > p[arg]) arg = v;
  assert.equal(arg, 4); // the +1 chain dominates
});

test("cache counts use the longest earlier match and never read the target", () => {
  //            0  1  2  3  4  5  6  7  8
  const hist = [9, 1, 2, 3, 7, 1, 2, 3, 5];
  const counts = new Float64Array(10);
  const touched: number[] = [];
  // history ends at 8 (target hist[8]=5 must not be visible): context [1,2,3] seen once before -> 7
  const total = cacheCounts(hist, 0, 8, 3, counts, touched);
  assert.equal(total, 1);
  assert.equal(counts[7], 1);
  assert.equal(counts[5], 0);
  const p = new Float64Array(10).fill(0.1);
  mixCache(p, counts, touched, total, 1);
  assert.ok(Math.abs(p.reduce((x, y) => x + y, 0) - 1) < 1e-12);
  assert.ok(Math.abs(p[7] - (0.05 + 0.5)) < 1e-12);
  // unseen context -> falls back to shorter suffixes, then 0
  assert.equal(cacheCounts([4, 4, 6], 0, 3, 3, counts, touched), 0);
});
