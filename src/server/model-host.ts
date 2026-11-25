/**
 * Loads the PC-resident network for the inference server.
 *
 * Resolution order (so the server never has to wait for training):
 *   1. models/release/{config.json, weights.bin}
 *   2. the newest models/<run>/ that has both files (by weights.bin mtime)
 *   3. a randomly initialised model of the benchmarked preset, with a loud warning
 *
 * Also precomputes what the lab asks for repeatedly: per-tensor statistics + histograms and a
 * fixed 3-D PCA basis of the residual stream (fit once on calibration text, so points do not jump
 * around between keystrokes).
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { loadModel, makeConfig, Transformer, PRESETS, type ModelConfig, type PresetName } from "../model/index.js";
import { tensorTable, noGrad, kernelBackend, type TensorInfo } from "../engine/index.js";
import { Tokenizer, BOF } from "../tokenizer/index.js";
import { fitPca, type PcaBasis } from "./pca.js";

export interface TensorStats {
  name: string;
  shape: number[];
  numel: number;
  offset: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  absMean: number;
  /** Fraction of |w| < 1e-3. */
  nearZero: number;
  /** 32-bin histogram over [min, max]. */
  hist: number[];
}

export type ModelSource =
  | { kind: "release"; dir: string }
  | { kind: "run"; dir: string; run: string }
  | { kind: "random"; reason: string };

export interface HostOptions {
  root?: string;
  modelsDir?: string;
  /** Force a specific model directory (overrides the resolution order). */
  modelDir?: string;
  /** Skip checkpoints and use random init (tests). */
  forceRandom?: boolean;
  log?: (msg: string) => void;
}

const CALIBRATION_JAVA = `package demo.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Simple in-memory cache with hit statistics. */
public class LruCache<K, V> {
    private final int capacity;
    private final Map<K, V> map = new HashMap<>();
    private final List<K> order = new ArrayList<>();
    private int hits = 0;

    public LruCache(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be positive");
        }
        this.capacity = capacity;
    }

    public V get(K key) {
        V value = map.get(key);
        if (value != null) {
            hits++;
            order.remove(key);
            order.add(key);
        }
        return value;
    }

    public void put(K key, V value) {
        if (map.size() >= capacity && !map.containsKey(key)) {
            K eldest = order.remove(0);
            map.remove(eldest);
        }
        map.put(key, value);
        order.add(key);
    }

    public static void main(String[] args) {
        LruCache<String, Integer> cache = new LruCache<>(2);
        for (int i = 0; i < 10; i++) {
            cache.put("k" + i, i);
            System.out.println(cache.get("k" + i));
        }
    }
}
`;

function tensorStats(info: TensorInfo, data: Float32Array): TensorStats {
  const w = data.subarray(info.offset, info.offset + info.numel);
  let sum = 0, sq = 0, abs = 0, mn = Infinity, mx = -Infinity, nz = 0;
  for (let i = 0; i < w.length; i++) {
    const v = w[i];
    sum += v; sq += v * v; abs += Math.abs(v);
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    if (Math.abs(v) < 1e-3) nz++;
  }
  const n = w.length;
  const mean = sum / n;
  const hist = new Array<number>(32).fill(0);
  const span = mx - mn || 1;
  for (let i = 0; i < n; i++) hist[Math.min(31, Math.floor(((w[i] - mn) / span) * 32))]++;
  return {
    name: info.name, shape: info.shape, numel: n, offset: info.offset,
    mean, std: Math.sqrt(Math.max(0, sq / n - mean * mean)), min: mn, max: mx, absMean: abs / n,
    nearZero: nz / n, hist,
  };
}

function isCheckpointDir(dir: string): boolean {
  return existsSync(join(dir, "config.json")) && existsSync(join(dir, "weights.bin"));
}

export class ModelHost {
  readonly model: Transformer;
  readonly tokenizer: Tokenizer;
  readonly source: ModelSource;
  readonly savedConfig: Record<string, unknown>;
  readonly tensors: TensorInfo[];
  readonly stats: TensorStats[];
  readonly statsByName = new Map<string, TensorStats>();
  readonly pca: PcaBasis;
  readonly warnings: string[] = [];
  readonly loadedAt = new Date().toISOString();

  constructor(opts: HostOptions = {}) {
    const log = opts.log ?? ((m: string) => console.log(m));
    const root = resolve(opts.root ?? process.cwd());
    const modelsDir = resolve(root, opts.modelsDir ?? "models");
    const tokPath = join(modelsDir, "tokenizer.json");
    if (!existsSync(tokPath)) throw new Error(`tokenizer not found at ${tokPath} (run \`npm run tokenizer\`)`);
    this.tokenizer = Tokenizer.fromJSON(readFileSync(tokPath, "utf8"));

    let model: Transformer | null = null;
    let source: ModelSource | null = null;
    let saved: Record<string, unknown> = {};
    const candidates: { dir: string; src: ModelSource }[] = [];
    if (!opts.forceRandom) {
      if (opts.modelDir) {
        const dir = resolve(root, opts.modelDir);
        const isRelease = dir === resolve(modelsDir, "release");
        candidates.push({ dir, src: isRelease ? { kind: "release", dir } : { kind: "run", dir, run: dir.split(/[\\/]/).pop() ?? dir } });
      } else {
        const rel = join(modelsDir, "release");
        if (isCheckpointDir(rel)) candidates.push({ dir: rel, src: { kind: "release", dir: rel } });
        if (existsSync(modelsDir)) {
          const runs = readdirSync(modelsDir, { withFileTypes: true })
            .filter((e) => e.isDirectory() && e.name !== "release" && isCheckpointDir(join(modelsDir, e.name)))
            .map((e) => ({ name: e.name, mtime: statSync(join(modelsDir, e.name, "weights.bin")).mtimeMs }))
            .sort((a, b) => b.mtime - a.mtime);
          for (const r of runs) {
            const dir = join(modelsDir, r.name);
            candidates.push({ dir, src: { kind: "run", dir, run: r.name } });
          }
        }
      }
    }
    for (const c of candidates) {
      try {
        const { model: m, config } = loadModel(c.dir);
        if (m.cfg.vocabSize !== this.tokenizer.vocabSize) {
          throw new Error(`vocab ${m.cfg.vocabSize} != tokenizer vocab ${this.tokenizer.vocabSize}`);
        }
        model = m;
        source = c.src;
        const { tensors: _t, ...rest } = config;
        saved = rest;
        break;
      } catch (e) {
        const msg = `skipping checkpoint ${c.dir}: ${(e as Error).message}`;
        this.warnings.push(msg);
        log(`[model] ${msg}`);
      }
    }
    if (!model) {
      const preset = readPreset(root);
      const cfg: ModelConfig = makeConfig(preset, this.tokenizer.vocabSize);
      model = new Transformer(cfg);
      model.init(1234);
      const reason = opts.forceRandom ? "forced random init" : "no trained checkpoint found in models/";
      source = { kind: "random", reason };
      saved = { format: "jal-v1", model: cfg, preset };
      const msg = `WARNING: ${reason}; serving a RANDOMLY INITIALISED '${preset}' model. Completions will be noise until models/release exists.`;
      this.warnings.push(msg);
      log(`[model] ${msg}`);
    }
    this.model = model;
    this.source = source!;
    this.savedConfig = saved;
    this.tensors = tensorTable(model.store);
    this.stats = this.tensors.map((t) => tensorStats(t, model!.store.data));
    for (const s of this.stats) this.statsByName.set(s.name, s);
    this.pca = this.fitResidualPca(root);
    log(`[model] ${describeSource(this.source)} | ${model.numParams.toLocaleString()} params | kernel=${kernelBackend()}`);
  }

  /** Fit one shared 3-D PCA basis over the residual stream of every layer (pooled). */
  private fitResidualPca(root: string): PcaBasis {
    const T = this.model.cfg.ctx;
    const seqs: number[][] = [];
    const valPath = join(root, "data", "val.bin");
    if (existsSync(valPath)) {
      const buf = readFileSync(valPath);
      const ids = new Uint16Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 2));
      const stride = Math.max(T, Math.floor(ids.length / 4));
      for (let s = 0; s < 4 && (s * stride + T) <= ids.length; s++) {
        seqs.push(Array.from(ids.subarray(s * stride, s * stride + T)));
      }
    }
    if (seqs.length === 0) {
      const ids = [BOF, ...this.tokenizer.encode(CALIBRATION_JAVA)];
      for (let s = 0; s + T <= ids.length && seqs.length < 4; s += T) seqs.push(ids.slice(s, s + T));
      if (seqs.length === 0) seqs.push(ids.slice(0, T));
    }
    const d = this.model.cfg.dModel;
    const rows: Float32Array[] = [];
    noGrad(() => {
      for (const seq of seqs) {
        const tr = this.model.forward(seq, 1, seq.length, { trace: true }).trace!;
        const mats = [tr.embed, ...tr.layers.map((l) => l.residOut)];
        for (const m of mats) for (let t = 0; t < seq.length; t++) rows.push(m.subarray(t * d, (t + 1) * d));
      }
    });
    return fitPca(rows, d, 3);
  }

  modelInfo() {
    const cfg = this.model.cfg;
    return {
      source: this.source,
      sourceLabel: describeSource(this.source),
      warnings: this.warnings,
      loadedAt: this.loadedAt,
      config: cfg,
      numParams: this.model.numParams,
      kernel: kernelBackend(),
      saved: this.savedConfig,
      tokenizer: {
        type: this.tokenizer.data.type,
        vocabSize: this.tokenizer.vocabSize,
        specials: { bof: 0, eof: 1, pad: 2 },
        pretokenRegex: this.tokenizer.data.pretoken.source,
        numMerges: this.tokenizer.data.merges.length,
      },
      pca: { components: this.pca.k, explained: this.pca.explained, fitRows: this.pca.n },
      tensors: this.stats,
    };
  }
}

function readPreset(root: string): PresetName {
  try {
    const bench = JSON.parse(readFileSync(join(root, "results", "bench.json"), "utf8")) as { choice?: { preset?: string } };
    const p = bench.choice?.preset;
    if (p && p in PRESETS) return p as PresetName;
  } catch { /* fall through */ }
  return "base";
}

export function describeSource(s: ModelSource): string {
  if (s.kind === "release") return "models/release";
  if (s.kind === "run") return `models/${s.run} (newest run)`;
  return `random init (${s.reason})`;
}
