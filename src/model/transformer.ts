/**
 * Decoder-only, pre-LN transformer (GPT-2 form):
 *
 *   h0 = tok_emb[x] + pos_emb[t]
 *   for each block l:  h = h + proj(attn(LN1(h)))      (causal multi-head self-attention)
 *                      h = h + fc2(gelu(fc1(LN2(h))))  (MLP)
 *   logits = LN_f(h) @ tok_emb^T                       (tied output head)
 *
 * `forward(..., { trace: true })` also returns every intermediate the web lab draws: the residual
 * stream before/after each sub-layer, attention probabilities per head, MLP neuron activations
 * (post-GELU), the final normalised residual and the logits. Tracing only keeps references to
 * tensors the forward pass computes anyway, so it costs no extra arithmetic.
 */
import {
  Tensor, ParamStore, Rng, add, causalAttention, crossEntropy, embedding, gelu, layernorm, linear,
  type IntArray, type ParamSpec, type ParamStoreOptions,
} from "../engine/index.js";
import { countParams, validateConfig, type ModelConfig } from "./config.js";

export interface LayerTrace {
  /** Residual stream entering the block [B*T, d]. */
  residIn: Float32Array;
  /** Residual after the attention sub-layer [B*T, d]. */
  residMid: Float32Array;
  /** Residual leaving the block [B*T, d]. */
  residOut: Float32Array;
  /** Attention probabilities [B, H, T, T] (query rows, key columns, causal). */
  attnProbs: Float32Array;
  /** Attention sub-layer output (after the output projection) [B*T, d]. */
  attnOut: Float32Array;
  /** MLP neuron activations after GELU [B*T, dMlp]. */
  mlpAct: Float32Array;
  /** MLP sub-layer output [B*T, d]. */
  mlpOut: Float32Array;
}

export interface ForwardTrace {
  B: number;
  T: number;
  tokEmb: Float32Array;
  posEmb: Float32Array;
  /** tok_emb + pos_emb [B*T, d]. */
  embed: Float32Array;
  layers: LayerTrace[];
  /** LN_f(residual) [B*T, d]. */
  final: Float32Array;
  /** [B*T, V]. */
  logits: Float32Array;
}

export interface ForwardOptions {
  /** Next-token targets [B*T]; if given, the mean cross-entropy is returned as `loss`. */
  targets?: IntArray;
  trace?: boolean;
}

export interface ForwardResult {
  logits: Tensor;
  loss?: Tensor;
  trace?: ForwardTrace;
}

export function paramSpecs(cfg: ModelConfig): ParamSpec[] {
  const { vocabSize: V, ctx: T, dModel: d, nLayers: L, dMlp: m } = cfg;
  const specs: ParamSpec[] = [
    { name: "tok_emb", shape: [V, d], decay: true },
    { name: "pos_emb", shape: [T, d], decay: false },
  ];
  for (let l = 0; l < L; l++) {
    const p = `blocks.${l}`;
    specs.push(
      { name: `${p}.ln1.g`, shape: [d], decay: false },
      { name: `${p}.ln1.b`, shape: [d], decay: false },
      { name: `${p}.attn.qkv.W`, shape: [3 * d, d], decay: true },
      { name: `${p}.attn.qkv.b`, shape: [3 * d], decay: false },
      { name: `${p}.attn.proj.W`, shape: [d, d], decay: true },
      { name: `${p}.attn.proj.b`, shape: [d], decay: false },
      { name: `${p}.ln2.g`, shape: [d], decay: false },
      { name: `${p}.ln2.b`, shape: [d], decay: false },
      { name: `${p}.mlp.fc.W`, shape: [m, d], decay: true },
      { name: `${p}.mlp.fc.b`, shape: [m], decay: false },
      { name: `${p}.mlp.proj.W`, shape: [d, m], decay: true },
      { name: `${p}.mlp.proj.b`, shape: [d], decay: false },
    );
  }
  specs.push({ name: "ln_f.g", shape: [d], decay: false }, { name: "ln_f.b", shape: [d], decay: false });
  return specs;
}

interface Block {
  ln1g: Tensor; ln1b: Tensor; qkvW: Tensor; qkvb: Tensor; projW: Tensor; projb: Tensor;
  ln2g: Tensor; ln2b: Tensor; fcW: Tensor; fcb: Tensor; fc2W: Tensor; fc2b: Tensor;
}

export class Transformer {
  readonly cfg: ModelConfig;
  readonly store: ParamStore;
  private readonly tokEmb: Tensor;
  private readonly posEmb: Tensor;
  private readonly blocks: Block[] = [];
  private readonly lnfG: Tensor;
  private readonly lnfB: Tensor;
  private posCache = new Map<string, Int32Array>();

  constructor(cfg: ModelConfig, storeOpts: ParamStoreOptions = {}) {
    validateConfig(cfg);
    this.cfg = { ...cfg };
    this.store = new ParamStore(paramSpecs(cfg), storeOpts);
    if (this.store.size !== countParams(cfg)) throw new Error("internal: param count mismatch");
    const s = this.store;
    this.tokEmb = s.get("tok_emb");
    this.posEmb = s.get("pos_emb");
    for (let l = 0; l < cfg.nLayers; l++) {
      const p = `blocks.${l}`;
      this.blocks.push({
        ln1g: s.get(`${p}.ln1.g`), ln1b: s.get(`${p}.ln1.b`),
        qkvW: s.get(`${p}.attn.qkv.W`), qkvb: s.get(`${p}.attn.qkv.b`),
        projW: s.get(`${p}.attn.proj.W`), projb: s.get(`${p}.attn.proj.b`),
        ln2g: s.get(`${p}.ln2.g`), ln2b: s.get(`${p}.ln2.b`),
        fcW: s.get(`${p}.mlp.fc.W`), fcb: s.get(`${p}.mlp.fc.b`),
        fc2W: s.get(`${p}.mlp.proj.W`), fc2b: s.get(`${p}.mlp.proj.b`),
      });
    }
    this.lnfG = s.get("ln_f.g");
    this.lnfB = s.get("ln_f.b");
  }

  get numParams(): number {
    return this.store.size;
  }

  /** GPT-2 initialisation: N(0, 0.02), residual projections scaled by 1/sqrt(2L), LN gain 1. */
  init(seed = 1234): void {
    const rng = new Rng(seed);
    const projStd = 0.02 / Math.sqrt(2 * this.cfg.nLayers);
    for (const e of this.store.entries) {
      const d = e.tensor.data;
      if (e.name.endsWith(".g")) d.fill(1);
      else if (e.name.endsWith(".b")) d.fill(0);
      else if (e.name.endsWith("attn.proj.W") || e.name.endsWith("mlp.proj.W")) rng.fillNormal(d, projStd);
      else if (e.name === "pos_emb") rng.fillNormal(d, 0.01);
      else rng.fillNormal(d, 0.02);
    }
  }

  private positions(B: number, T: number): Int32Array {
    const key = `${B}x${T}`;
    let p = this.posCache.get(key);
    if (!p) {
      p = new Int32Array(B * T);
      for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) p[b * T + t] = t;
      this.posCache.set(key, p);
    }
    return p;
  }

  /** ids: [B*T] token ids (row-major by sequence). */
  forward(ids: IntArray, B: number, T: number, opts: ForwardOptions = {}): ForwardResult {
    const { cfg } = this;
    if (ids.length !== B * T) throw new Error(`forward: ${ids.length} ids for B=${B}, T=${T}`);
    if (T > cfg.ctx) throw new Error(`forward: T=${T} exceeds context ${cfg.ctx}`);
    const tok = embedding(ids, this.tokEmb, [B * T]);
    const pos = embedding(this.positions(B, T), this.posEmb, [B * T]);
    let h = add(tok, pos);
    const layers: LayerTrace[] = [];
    for (const blk of this.blocks) {
      const residIn = h;
      const a = layernorm(h, blk.ln1g, blk.ln1b);
      const qkv = linear(a, blk.qkvW, blk.qkvb);
      const att = causalAttention(qkv, B, T, cfg.nHeads);
      const attnOut = linear(att.out, blk.projW, blk.projb);
      h = add(h, attnOut);
      const residMid = h;
      const m = layernorm(h, blk.ln2g, blk.ln2b);
      const act = gelu(linear(m, blk.fcW, blk.fcb));
      const mlpOut = linear(act, blk.fc2W, blk.fc2b);
      h = add(h, mlpOut);
      if (opts.trace) {
        layers.push({
          residIn: residIn.data, residMid: residMid.data, residOut: h.data, attnProbs: att.probs,
          attnOut: attnOut.data, mlpAct: act.data, mlpOut: mlpOut.data,
        });
      }
    }
    const f = layernorm(h, this.lnfG, this.lnfB);
    const logits = linear(f, this.tokEmb, null);
    const result: ForwardResult = { logits };
    if (opts.targets) result.loss = crossEntropy(logits, opts.targets);
    if (opts.trace) {
      result.trace = {
        B, T, tokEmb: tok.data, posEmb: pos.data, embed: layers[0]?.residIn ?? h.data, layers,
        final: f.data, logits: logits.data,
      };
    }
    return result;
  }
}
