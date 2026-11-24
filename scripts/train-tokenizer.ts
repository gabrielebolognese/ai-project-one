/**
 * Tokenizer trainer + corpus encoder:  npm run tokenizer [-- --vocab 2048]
 *
 * Trains byte-level BPE on the TRAIN split only (data/corpus/train.jsonl), saves
 * models/tokenizer.json, encodes every split to data/{train,val,test}.bin (raw LE uint16, each
 * file wrapped in <|bof|> ... <|eof|>), and writes data/stats.json + data/meta.json.
 * Every file of every split is round-trip checked (decode(encode(x)) === x).
 * Also measures bytes/token for smaller vocabularies (merges are nested, so one run is enough).
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { countPretokens, Tokenizer, trainBpe, EOF, BOF, MERGE_BASE, type TokenizerData } from "../src/tokenizer/index.js";

const ROOT = resolve(import.meta.dirname, "..");
const DATA = join(ROOT, "data");
const { values: a } = parseArgs({ options: { vocab: { type: "string", default: "2048" } } });
const VOCAB = Number(a.vocab);

interface FileRow {
  repo: string;
  path: string;
  text: string;
}
const readSplit = (s: string): FileRow[] =>
  readFileSync(join(DATA, "corpus", `${s}.jsonl`), "utf8")
    .split("\n")
    .filter(Boolean)
    .map((l) => JSON.parse(l) as FileRow);

const t0 = Date.now();
const log = (m: string) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1)}s] ${m}`);

if (!existsSync(join(DATA, "corpus", "train.jsonl"))) throw new Error("run `npm run corpus` first");
const train = readSplit("train");
const val = readSplit("val");
const test = readSplit("test");
log(`loaded train ${train.length} / val ${val.length} / test ${test.length} files`);

// ---- train BPE on the train split only; keep 4096 merges so smaller vocabs can be evaluated too ----
const sweepVocabs = [1024, 2048, 4096].filter((v) => v !== VOCAB);
const trainVocab = Math.max(VOCAB, 4096);
const counts = countPretokens(train.map((f) => f.text));
log(`${counts.size} distinct pre-tokens`);
const full = trainBpe(counts, trainVocab, log);
const trainBytes = train.reduce((s, f) => s + Buffer.byteLength(f.text), 0);

function truncate(d: TokenizerData, vocab: number): TokenizerData {
  const merges = d.merges.slice(0, vocab - MERGE_BASE);
  return { ...d, vocabSize: MERGE_BASE + merges.length, merges, tokens: d.tokens.slice(0, MERGE_BASE + merges.length) };
}
const data = truncate(full, VOCAB);
data.trainInfo = {
  corpus: "train split only (data/corpus/train.jsonl)",
  train_files: train.length,
  train_bytes: trainBytes,
  distinct_pretokens: counts.size,
  trained_at: new Date().toISOString(),
};
mkdirSync(join(ROOT, "models"), { recursive: true });
const tok = new Tokenizer(data);
writeFileSync(join(ROOT, "models", "tokenizer.json"), tok.toJSON());
log(`saved models/tokenizer.json (vocab ${tok.vocabSize})`);

// ---- encode splits ----
function encodeSplit(name: string, rows: FileRow[], t: Tokenizer) {
  const chunks: Uint16Array[] = [];
  let tokens = 0;
  let bytes = 0;
  let roundTripFail = 0;
  const perRepo = new Map<string, { files: number; tokens: number; bytes: number }>();
  for (const r of rows) {
    const ids = t.encode(r.text, { bof: true, eof: true });
    if (t.decode(ids) !== r.text) roundTripFail++;
    chunks.push(Uint16Array.from(ids));
    const b = Buffer.byteLength(r.text);
    tokens += ids.length;
    bytes += b;
    const p = perRepo.get(r.repo) ?? { files: 0, tokens: 0, bytes: 0 };
    p.files++;
    p.tokens += ids.length;
    p.bytes += b;
    perRepo.set(r.repo, p);
  }
  const all = new Uint16Array(tokens);
  let o = 0;
  for (const c of chunks) (all.set(c, o), (o += c.length));
  return { name, all, tokens, bytes, roundTripFail, perRepo: Object.fromEntries(perRepo) };
}

const enc = {
  train: encodeSplit("train", train, tok),
  val: encodeSplit("val", val, tok),
  test: encodeSplit("test", test, tok),
};
for (const [name, e] of Object.entries(enc)) {
  writeFileSync(join(DATA, `${name}.bin`), Buffer.from(e.all.buffer, e.all.byteOffset, e.all.byteLength));
  log(`${name}: ${e.tokens} tokens, ${(e.bytes / e.tokens).toFixed(3)} bytes/token, round-trip failures ${e.roundTripFail}`);
  if (e.roundTripFail) throw new Error(`${name}: ${e.roundTripFail} files do not round-trip`);
}
let maxId = 0;
for (const e of Object.values(enc)) for (let i = 0; i < e.all.length; i++) if (e.all[i] > maxId) maxId = e.all[i];
if (maxId >= tok.vocabSize) throw new Error(`token id ${maxId} out of range`);

// ---- bytes/token for other vocab sizes on val + test (merges are nested prefixes) ----
const sweep: Record<string, { val_bytes_per_token: number; test_bytes_per_token: number; embedding_params_d128: number }> = {};
for (const v of [...sweepVocabs, VOCAB].sort((x, y) => x - y)) {
  const t = new Tokenizer(truncate(full, v));
  const bpt = (rows: FileRow[]) => {
    let n = 0;
    let b = 0;
    for (const r of rows) (n += t.encode(r.text).length, (b += Buffer.byteLength(r.text)));
    return +(b / n).toFixed(4);
  };
  sweep[v] = { val_bytes_per_token: bpt(val), test_bytes_per_token: bpt(test), embedding_params_d128: v * 128 };
  log(`vocab ${v}: val ${sweep[v].val_bytes_per_token} B/tok, test ${sweep[v].test_bytes_per_token} B/tok`);
}

const files = train.length + val.length + test.length;
const stats = {
  files,
  train_files: train.length,
  val_files: val.length,
  test_files: test.length,
  train_tokens: enc.train.tokens,
  val_tokens: enc.val.tokens,
  test_tokens: enc.test.tokens,
  vocab_size: tok.vocabSize,
  train_bytes: enc.train.bytes,
  val_bytes: enc.val.bytes,
  test_bytes: enc.test.bytes,
  bytes_per_token: {
    train: +(enc.train.bytes / enc.train.tokens).toFixed(4),
    val: +(enc.val.bytes / enc.val.tokens).toFixed(4),
    test: +(enc.test.bytes / enc.test.tokens).toFixed(4),
  },
  round_trip_exact: true,
  max_token_id: maxId,
  special_tokens: { bof: BOF, eof: EOF },
  repos: { train: enc.train.perRepo, val: enc.val.perRepo, test: enc.test.perRepo },
  vocab_sweep: sweep,
  note: "Tokenizer trained on the train split only. Splits are by repository (see data/SOURCES.md).",
};
writeFileSync(join(DATA, "stats.json"), JSON.stringify(stats, null, 1));
writeFileSync(join(DATA, "meta.json"), JSON.stringify({ vocabSize: tok.vocabSize }));
mkdirSync(join(ROOT, "results"), { recursive: true });
writeFileSync(join(ROOT, "results", "corpus-stats.json"), JSON.stringify(stats, null, 1));
log(`wrote data/stats.json: train ${stats.train_tokens} / val ${stats.val_tokens} / test ${stats.test_tokens} tokens`);
