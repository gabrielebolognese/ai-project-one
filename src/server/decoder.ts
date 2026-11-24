/**
 * Incremental (KV-cached) decoder for generation.
 *
 * Re-running Transformer.forward over the whole window for every generated token costs ~T times
 * more than needed (measured ~87 ms/token at T = 128 on the 5700U). This decoder caches each
 * layer's keys and values, so one new token costs one row of every matmul plus attention over the
 * cache. It reuses the engine's own ops (embedding, layernorm, linear, gelu, add, causalAttention)
 * under noGrad. Only the single-query attention against the cache is written here.
 * decoder.test.ts checks its logits against Transformer.forward.
 *
 * Positions are absolute (learned table of size ctx), so once the cache is full the window must be
 * rebuilt. `generate` then re-prefills the last `ctx - slide` tokens (context shift), which buys
 * `slide` cheap steps per rebuild. That is the only place generation sees slightly less context
 * than a full-window forward would.
 */
import { Tensor, add, causalAttention, embedding, gelu, layernorm, linear, noGrad } from "../engine/index.js";
import type { Transformer } from "../model/index.js";

interface LayerW {
  ln1g: Tensor; ln1b: Tensor; qkvW: Tensor; qkvb: Tensor; projW: Tensor; projb: Tensor;
  ln2g: Tensor; ln2b: Tensor; fcW: Tensor; fcb: Tensor; fc2W: Tensor; fc2b: Tensor;
}

export class IncrementalDecoder {
  readonly ctx: number;
  private readonly D: number;
  private readonly H: number;
  private readonly hd: number;
  private readonly layers: LayerW[] = [];
  private readonly tokEmb: Tensor;
  private readonly posEmb: Tensor;
  private readonly lnfG: Tensor;
  private readonly lnfB: Tensor;
  private readonly kCache: Float32Array[];
  private readonly vCache: Float32Array[];
  private readonly scores: Float32Array;
  len = 0;

  constructor(model: Transformer) {
    const { ctx, dModel: D, nHeads: H, nLayers: L } = model.cfg;
    this.ctx = ctx; this.D = D; this.H = H; this.hd = D / H;
    const s = model.store;
    this.tokEmb = s.get("tok_emb");
    this.posEmb = s.get("pos_emb");
    this.lnfG = s.get("ln_f.g");
    this.lnfB = s.get("ln_f.b");
    for (let l = 0; l < L; l++) {
      const p = `blocks.${l}`;
      this.layers.push({
        ln1g: s.get(`${p}.ln1.g`), ln1b: s.get(`${p}.ln1.b`),
        qkvW: s.get(`${p}.attn.qkv.W`), qkvb: s.get(`${p}.attn.qkv.b`),
        projW: s.get(`${p}.attn.proj.W`), projb: s.get(`${p}.attn.proj.b`),
        ln2g: s.get(`${p}.ln2.g`), ln2b: s.get(`${p}.ln2.b`),
        fcW: s.get(`${p}.mlp.fc.W`), fcb: s.get(`${p}.mlp.fc.b`),
        fc2W: s.get(`${p}.mlp.proj.W`), fc2b: s.get(`${p}.mlp.proj.b`),
      });
    }
    this.kCache = this.layers.map(() => new Float32Array(ctx * D));
    this.vCache = this.layers.map(() => new Float32Array(ctx * D));
    this.scores = new Float32Array(ctx);
  }

  get full(): boolean {
    return this.len >= this.ctx;
  }

  private headLogits(lastRow: Float32Array): Float32Array {
    const x = new Tensor(lastRow, [1, this.D]);
    const f = layernorm(x, this.lnfG, this.lnfB);
    return linear(f, this.tokEmb, null).data;
  }

  /** Reset the cache and run the prompt (len <= ctx). Returns next-token logits [V]. */
  prefill(ids: ArrayLike<number>): Float32Array {
    const T = ids.length;
    if (T < 1 || T > this.ctx) throw new Error(`prefill: need 1..${this.ctx} tokens, got ${T}`);
    const D = this.D;
    return noGrad(() => {
      const pos = new Int32Array(T);
      for (let t = 0; t < T; t++) pos[t] = t;
      let h = add(embedding(Array.from(ids), this.tokEmb, [T]), embedding(pos, this.posEmb, [T]));
      this.layers.forEach((w, l) => {
        const qkv = linear(layernorm(h, w.ln1g, w.ln1b), w.qkvW, w.qkvb);
        const q = qkv.data, kc = this.kCache[l], vc = this.vCache[l];
        for (let t = 0; t < T; t++) {
          kc.set(q.subarray(t * 3 * D + D, t * 3 * D + 2 * D), t * D);
          vc.set(q.subarray(t * 3 * D + 2 * D, t * 3 * D + 3 * D), t * D);
        }
        const att = causalAttention(qkv, 1, T, this.H);
        h = add(h, linear(att.out, w.projW, w.projb));
        h = add(h, linear(gelu(linear(layernorm(h, w.ln2g, w.ln2b), w.fcW, w.fcb)), w.fc2W, w.fc2b));
      });
      this.len = T;
      return this.headLogits(h.data.slice((T - 1) * D, T * D));
    });
  }

  /** Append one token (cache must not be full). Returns next-token logits [V]. */
  step(id: number): Float32Array {
    if (this.len >= this.ctx) throw new Error("step: cache full, re-prefill a shorter window");
    const { D, H, hd } = this;
    const t = this.len;
    const sc = 1 / Math.sqrt(hd);
    return noGrad(() => {
      const x0 = new Float32Array(D);
      const te = this.tokEmb.data, pe = this.posEmb.data;
      for (let j = 0; j < D; j++) x0[j] = te[id * D + j] + pe[t * D + j];
      let h = new Tensor(x0, [1, D]);
      this.layers.forEach((w, l) => {
        const qkv = linear(layernorm(h, w.ln1g, w.ln1b), w.qkvW, w.qkvb).data;
        const kc = this.kCache[l], vc = this.vCache[l];
        kc.set(qkv.subarray(D, 2 * D), t * D);
        vc.set(qkv.subarray(2 * D, 3 * D), t * D);
        const o = new Float32Array(D);
        const s = this.scores;
        for (let hh = 0; hh < H; hh++) {
          const c0 = hh * hd;
          let m = -Infinity;
          for (let j = 0; j <= t; j++) {
            let dot = 0;
            const ko = j * D + c0;
            for (let c = 0; c < hd; c++) dot += qkv[c0 + c] * kc[ko + c];
            const v = dot * sc;
            s[j] = v;
            if (v > m) m = v;
          }
          let z = 0;
          for (let j = 0; j <= t; j++) { const e = Math.exp(s[j] - m); s[j] = e; z += e; }
          const inv = 1 / z;
          for (let j = 0; j <= t; j++) {
            const p = s[j] * inv;
            const vo = j * D + c0;
            for (let c = 0; c < hd; c++) o[c0 + c] += p * vc[vo + c];
          }
        }
        h = add(h, linear(new Tensor(o, [1, D]), w.projW, w.projb));
        h = add(h, linear(gelu(linear(layernorm(h, w.ln2g, w.ln2b), w.fcW, w.fcb)), w.fc2W, w.fc2b));
      });
      this.len = t + 1;
      return this.headLogits(h.data);
    });
  }
}
