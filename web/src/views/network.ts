/**
 * The live "Network" scene. Everything drawn here comes from a real traced forward pass of the
 * local model (WS /api/trace) plus the real token/position embedding tables (/api/weights).
 *
 * Tower mode (bottom -> top, the way the residual stream flows):
 *   tokens -> embedding columns (E[id] + P[t], 128 cells each) -> for each layer: attention arcs
 *   (opacity/width = attention probability, colour = head) and the MLP neuron grid (dots light up
 *   with the post-GELU activation) -> unembedding -> softmax bars (right column).
 *   The residual stream is a river per token whose width is the measured L2 norm at each stage.
 * Heads mode: a close-up of one layer, one panel per head with arcs and the full attention matrix.
 */
import type { CompleteResult, ModelConfig, StepInfo, TokensMsg, TracePayload } from "../api";
import { C, DIV_LUT, FONT_MONO, FONT_SANS, FONT_SERIF, HEAD_COLORS, divIndex, fmt, pct, seqColor, showToken, withAlpha } from "../theme";
import { TweenNum, TweenVec, bezierPartial, clamp01, easeInOutCubic, easeOutCubic, fitCanvas } from "../tween";

interface TokItem {
  key: number; // absolute position in the file's token stream (truncated + index)
  idx: number; // index inside the current prompt window
  id: number;
  text: string;
  x: TweenNum;
  alpha: TweenNum;
  emb: TweenVec;
  norms: TweenVec;
  attn: TweenVec; // [L*H]: attention from the focus query to this key
  pca: TweenVec; // [S*3]
  alive: boolean;
}

interface Bar {
  id: number;
  text: string;
  p: TweenNum;
  rank: TweenNum;
  alpha: TweenNum;
}

interface Layout {
  w: number;
  h: number;
  towerX0: number; // tokens area left
  towerX1: number; // tokens area right
  gutterX: number;
  rightX: number;
  rightW: number;
  rows: { y: number; h: number }[]; // index = row; 0 tokens, 1 embed, 2+2l attn, 3+2l mlp, last = output
  slots: number;
  slotW: number;
}

export type ViewMode = "tower" | "heads";

export interface NetworkCallbacks {
  onHoverToken(range: { from: number; to: number } | null): void;
  onCaption(html: string): void;
  onFocusChange(label: string): void;
}

const STAGE_MS = 900;

export class NetworkView {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private visCtx: CanvasRenderingContext2D;
  private cache = document.createElement("canvas");
  private cacheCtx: CanvasRenderingContext2D;
  private animUntil = 0;
  private cacheValid = false;
  renders = 0;
  private cfg: ModelConfig | null = null;
  private tokEmb: Float32Array | null = null;
  private posEmb: Float32Array | null = null;
  private trace: TracePayload | null = null;
  private tokens: TokensMsg | null = null;
  private items = new Map<number, TokItem>();
  private order: number[] = [];
  private focusKey: number | null = null; // null = follow the last token
  private mlpVec = new TweenVec(0);
  private mlpScale = new TweenNum(1);
  private embScale = new TweenNum(1);
  private normScale = new TweenNum(1);
  private pcaScale = new TweenNum(1);
  private headMats = new TweenVec(0);
  private bars = new Map<number, Bar>();
  private lens: { text: string; p: number; t0: number }[] = [];
  private steps: { s: StepInfo; at: number }[] = [];
  private completion: CompleteResult | null = null;
  private pulseT0 = -1e9;
  private stage = new TweenNum(99, 650);
  private replaying = false;
  private replayTimer: number | null = null;
  private tokenReplayTimer: number | null = null;
  private layout: Layout | null = null;
  private mouse: { x: number; y: number } | null = null;
  private hoverKey: number | null = null;
  private lastFrame = 0;
  private flowT = 0;
  mode: ViewMode = "tower";
  selLayer = -1;
  selHead = -1;
  allPairs = true;
  entropyOf: number[] = [];

  constructor(private host: HTMLElement, private cb: NetworkCallbacks) {
    this.canvas = document.createElement("canvas");
    this.canvas.className = "net-canvas";
    host.appendChild(this.canvas);
    this.visCtx = this.canvas.getContext("2d")!;
    this.cacheCtx = this.cache.getContext("2d")!;
    this.ctx = this.cacheCtx;
    this.canvas.addEventListener("mousemove", (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
    });
    this.canvas.addEventListener("mouseleave", () => {
      this.mouse = null;
      if (this.hoverKey !== null) { this.hoverKey = null; this.cb.onHoverToken(null); }
    });
    this.canvas.addEventListener("click", () => this.onClick());
  }

  setModel(cfg: ModelConfig, tokEmb: Float32Array | null, posEmb: Float32Array | null): void {
    this.cfg = cfg;
    this.tokEmb = tokEmb;
    this.posEmb = posEmb;
    this.bump();
  }

  setTokens(m: TokensMsg): void {
    this.tokens = m;
  }

  get hasTrace(): boolean {
    return this.trace !== null;
  }

  get stageCount(): number {
    return this.cfg ? 3 + 2 * this.cfg.nLayers : 0; // rows: tokens, embed, L*(attn,mlp), output
  }

  /** New real trace from the server: retarget every tween. */
  setTrace(tr: TracePayload): void {
    const now = performance.now();
    this.trace = tr;
    if (!this.tokens) return; // tokens always arrive paired with their trace (api.ts)
    if (this.focusKey !== null && !this.items.has(this.focusKey)) this.focusKey = null;
    this.retarget(now, true);
    this.pulseT0 = now;
    if (!this.replaying) this.stage.jump(99);
    this.bump(1300);
    this.updateCaption();
  }

  setCompletion(c: CompleteResult | null): void {
    const now = performance.now();
    this.completion = c;
    this.steps = c ? c.steps.map((s, i) => ({ s, at: now + i * 90 })) : [];
    this.bump(this.steps.length * 90 + 600);
  }

  focusLabel(): string {
    if (this.focusKey === null) return "following the last token";
    const it = this.items.get(this.focusKey);
    return it ? `query = '${showToken(it.text)}' (pos ${it.idx})` : "following the last token";
  }

  setFocus(key: number | null): void {
    this.focusKey = key;
    this.retarget(performance.now(), false);
    this.bump();
    this.cb.onFocusChange(this.focusLabel());
    this.updateCaption();
  }

  // ---------------------------------------------------------------------------------------------
  // targets

  private focusIdx(): number {
    const tr = this.trace!;
    if (this.focusKey === null) return tr.T - 1;
    const it = this.items.get(this.focusKey);
    return it ? it.idx : tr.T - 1;
  }

  private attnWeight(l: number, h: number, q: number, k: number): number {
    const tr = this.trace!;
    if (k > q) return 0;
    if (q === tr.T - 1) return tr.attention.last[l][h][k] ?? 0;
    const w0 = tr.attention.windowStart, W = tr.attention.window;
    if (q < w0 || k < w0) return 0;
    return tr.attention.matrices[l][h][(q - w0) * W + (k - w0)] ?? 0;
  }

  private retarget(now: number, newTrace: boolean): void {
    const tr = this.trace, tk = this.tokens, cfg = this.cfg;
    if (!tr || !tk || !cfg) return;
    const L = tr.nLayers, H = tr.nHeads, S = tr.stages.length, d = tr.dModel, m = tr.dMlp;
    const lay = this.computeLayout();
    const Wd = Math.min(tr.attention.window, lay.slots, tr.T);
    const first = tr.T - Wd;
    const keys: number[] = [];
    for (let i = first; i < tr.T; i++) keys.push(tk.truncated + i);
    for (const it of this.items.values()) it.alive = false;
    const fq = this.focusIdx();
    let embAbs = 0, embN = 0, normMax = 0;
    const pcaAbs: number[] = [];
    keys.forEach((key, slot) => {
      const i = key - tk.truncated;
      const tv = tk.tokens[i] ?? { id: 0, text: "?" };
      let it = this.items.get(key);
      const x = lay.towerX0 + (slot + 0.5) * lay.slotW;
      if (!it) {
        it = {
          key, idx: i, id: tv.id, text: tv.text, x: new TweenNum(x + lay.slotW * 0.8), alpha: new TweenNum(0),
          emb: new TweenVec(d), norms: new TweenVec(S), attn: new TweenVec(L * H), pca: new TweenVec(S * 3), alive: true,
        };
        this.items.set(key, it);
      }
      it.alive = true;
      it.idx = i;
      it.id = tv.id;
      it.text = tv.text;
      it.x.set(x, now, 500);
      it.alpha.set(1, now, 450);
      // embedding column: E[id] + P[pos], straight from the weight tables
      const e = new Float32Array(d);
      if (this.tokEmb && this.posEmb) {
        for (let j = 0; j < d; j++) e[j] = this.tokEmb[tv.id * d + j] + this.posEmb[i * d + j];
      }
      for (let j = 0; j < d; j++) { embAbs += e[j] * e[j]; embN++; }
      it.emb.set(e, now);
      const nr = new Float32Array(S);
      const pc = new Float32Array(S * 3);
      for (let s = 0; s < S; s++) {
        nr[s] = tr.norms[s][i];
        normMax = Math.max(normMax, nr[s]);
        for (let c = 0; c < 3; c++) {
          pc[s * 3 + c] = tr.pca[s][i * 3 + c];
          if (c < 2) pcaAbs.push(Math.abs(pc[s * 3 + c]));
        }
      }
      it.norms.set(nr, now);
      it.pca.set(pc, now);
      const at = new Float32Array(L * H);
      for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) at[l * H + h] = this.attnWeight(l, h, fq, i);
      it.attn.set(at, now, newTrace ? 550 : 420);
    });
    for (const it of this.items.values()) {
      if (!it.alive) {
        it.alpha.set(0, now, 350);
        it.x.set(it.x.target - lay.slotW, now, 450);
        const z = new Float32Array(L * H);
        it.attn.set(z, now, 300);
      }
    }
    // drop items that are invisible and old
    for (const [k, it] of this.items) if (!it.alive && it.alpha.get(now) < 0.01 && !it.alpha.active(now)) this.items.delete(k);
    this.order = keys;
    this.embScale.set(embN ? 2.2 * Math.sqrt(embAbs / embN) : 1, now);
    this.normScale.set(normMax || 1, now);
    pcaAbs.sort((x, y) => x - y);
    this.pcaScale.set(pcaAbs[Math.floor(pcaAbs.length * 0.8)] || 1, now);

    // MLP neurons of the focus token (full vector for the last token, top-8 per token otherwise)
    const mv = new Float32Array(L * m);
    let mmax = 0;
    for (let l = 0; l < L; l++) {
      if (fq === tr.T - 1 && tr.mlp.lastAll) {
        for (let n = 0; n < m; n++) mv[l * m + n] = tr.mlp.lastAll[l][n];
      } else if (fq === tr.T - 1) {
        for (const { i, v } of tr.mlp.lastTop[l]) mv[l * m + i] = v;
      } else if (tr.mlp.windowTop && fq >= tr.attention.windowStart) {
        for (const { i, v } of tr.mlp.windowTop[l][fq - tr.attention.windowStart]) mv[l * m + i] = v;
      } else {
        mv[l * m + tr.mlp.argmax[l][fq]] = tr.mlp.argmaxV[l][fq];
      }
    }
    for (let i = 0; i < mv.length; i++) mmax = Math.max(mmax, mv[i]);
    this.mlpVec.set(mv, now);
    this.mlpScale.set(mmax || 1, now);

    // full attention matrices of the selected layer (heads close-up)
    const lsel = this.selLayer >= 0 ? this.selLayer : L - 1;
    const W = tr.attention.window;
    const hm = new Float32Array(H * W * W);
    for (let h = 0; h < H; h++) hm.set(tr.attention.matrices[lsel][h], h * W * W);
    this.headMats.set(hm, now);

    if (newTrace) {
      // softmax bars, keyed by token id so a bar slides to its new rank
      const seen = new Set<number>();
      tr.next.topk.forEach((t, r) => {
        seen.add(t.id);
        let b = this.bars.get(t.id);
        if (!b) {
          b = { id: t.id, text: t.text, p: new TweenNum(0), rank: new TweenNum(r + 0.6), alpha: new TweenNum(0) };
          this.bars.set(t.id, b);
        }
        b.p.set(t.p, now, 600);
        b.rank.set(r, now, 600);
        b.alpha.set(1, now, 400);
      });
      for (const [id, b] of this.bars) {
        if (!seen.has(id)) {
          b.alpha.set(0, now, 300);
          b.p.set(0, now, 400);
          if (b.alpha.get(now) < 0.01) this.bars.delete(id);
        }
      }
      // logit lens per stage (last token)
      this.lens = tr.logitLens.map((x, s) => {
        const t = x.topk[0];
        const prev = this.lens[s];
        return { text: t.text, p: t.p, t0: prev && prev.text === t.text ? prev.t0 : now };
      });
      this.entropyOf = tr.logitLens.map((x) => x.entropy);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // replay controls

  playPass(): void {
    this.stopReplay();
    const n = this.stageCount;
    if (!n || !this.trace) return;
    this.replaying = true;
    this.stage.jump(-0.3);
    let r = 0;
    const tick = () => {
      this.stage.set(r, performance.now(), 650);
      this.updateCaption(r);
      r++;
      if (r < n) this.replayTimer = window.setTimeout(tick, STAGE_MS);
      else this.replayTimer = window.setTimeout(() => { this.replaying = false; this.stage.set(99, performance.now(), 400); this.updateCaption(); }, STAGE_MS * 1.6);
    };
    tick();
  }

  stepPass(dir: 1 | -1): void {
    this.stopReplay(false);
    const n = this.stageCount;
    if (!n) return;
    this.replaying = true;
    const cur = Math.round(Math.min(this.stage.target, n - 1));
    const next = Math.max(0, Math.min(n - 1, (this.stage.target >= 99 ? (dir > 0 ? -1 : n) : cur) + dir));
    this.stage.set(next, performance.now(), 600);
    this.updateCaption(next);
  }

  playTokens(): void {
    this.stopReplay();
    if (!this.trace) return;
    const keys = [...this.order];
    let i = 0;
    const tick = () => {
      if (i >= keys.length) { this.setFocus(null); this.tokenReplayTimer = null; return; }
      this.setFocus(keys[i] === keys[keys.length - 1] ? null : keys[i]);
      i++;
      this.tokenReplayTimer = window.setTimeout(tick, 650);
    };
    tick();
  }

  stopReplay(resetStage = true): void {
    if (this.replayTimer) clearTimeout(this.replayTimer);
    if (this.tokenReplayTimer) clearTimeout(this.tokenReplayTimer);
    this.replayTimer = this.tokenReplayTimer = null;
    if (resetStage) {
      this.replaying = false;
      this.stage.set(99, performance.now(), 300);
    }
  }

  setLayer(l: number): void {
    this.selLayer = l;
    this.retarget(performance.now(), false);
    this.bump();
    this.updateCaption();
  }

  setHead(h: number): void {
    this.selHead = h;
    this.bump();
    this.updateCaption();
  }

  // ---------------------------------------------------------------------------------------------
  // narration (generated from the real numbers of this trace)

  private updateCaption(row?: number): void {
    const tr = this.trace, cfg = this.cfg;
    if (!tr || !cfg || !this.tokens) return;
    const L = tr.nLayers, H = tr.nHeads;
    const n = this.stageCount;
    const r = row ?? n - 1;
    const fq = this.focusIdx();
    const q = this.tokens.tokens[fq];
    const qs = `<span class="tk">${esc(showToken(q?.text ?? "?"))}</span>`;
    const num = (i: number) => `<b>${i}</b>`;
    let html = "";
    if (r === 0) {
      html = `<i>Tokenize.</i> The ${num(this.tokens.prefixChars)} characters before the cursor become ${num(tr.T + this.tokens.truncated)} byte-level BPE tokens (vocab ${cfg.vocabSize}); the model sees the last ${num(tr.T)} (context ${cfg.ctx}).`;
    } else if (r === 1) {
      html = `<i>Embed.</i> Each token id selects a row of <span class="m">E</span> (${cfg.vocabSize}×${cfg.dModel}) and adds a learned position vector: <span class="m">x₀ = E[id] + P[t]</span>. Each column of cells is those ${cfg.dModel} numbers (blue +, red −).`;
    } else if (r < n - 1) {
      const l = Math.floor((r - 2) / 2);
      if ((r - 2) % 2 === 0) {
        let best = { h: 0, k: fq, w: 0 };
        for (let h = 0; h < H; h++) for (let k = 0; k <= fq; k++) {
          const w = this.attnWeight(l, h, fq, k);
          if (w > best.w) best = { h, k, w };
        }
        const kt = this.tokens.tokens[best.k];
        const dist = fq - best.k;
        html = `<i>Layer ${l + 1} attention.</i> Query ${qs} mixes in earlier tokens with <span class="m">softmax(QKᵀ/√${cfg.dModel / H})V</span>. ` +
          `<span style="color:${HEAD_COLORS[best.h]}">Head ${best.h + 1}</span> puts ${num(Math.round(best.w * 100))}% of its weight on <span class="tk">${esc(showToken(kt?.text ?? "?"))}</span> ` +
          `(${dist === 0 ? "itself" : `${dist} token${dist > 1 ? "s" : ""} back`}).`;
      } else {
        const frac = tr.mlp.activeFrac[l];
        const top = tr.mlp.lastTop[l][0];
        const lens = tr.logitLens[2 + 2 * l]?.topk[0];
        html = `<i>Layer ${l + 1} MLP.</i> ${num(Math.round(frac * 100))}% of the ${cfg.dMlp} GELU neurons are positive for the last token; ` +
          `neuron <b>#${top?.i}</b> is the strongest (${fmt(top?.v ?? 0, 2)}). Read through the unembedding, the residual now says ` +
          `<span class="tk">${esc(showToken(lens?.text ?? "?"))}</span> (${pct(lens?.p ?? 0, 0)}).`;
      }
    } else {
      const t = tr.next.topk[0];
      html = `<i>Unembed.</i> <span class="m">p = softmax(LN(x)·Eᵀ)</span> over ${cfg.vocabSize} tokens. Top guess <span class="tk">${esc(showToken(t.text))}</span> at ${num(Math.round(t.p * 100))}%, ` +
        `entropy ${fmt(tr.next.entropy, 2)} nats. Trace ${tr.ms} ms on the CPU.`;
    }
    void L;
    this.cb.onCaption(html);
  }

  // ---------------------------------------------------------------------------------------------
  // layout

  private computeLayout(): Layout {
    const { w, h } = { w: this.canvas.clientWidth || 800, h: this.canvas.clientHeight || 600 };
    if (this.layout && this.layout.w === w && this.layout.h === h) return this.layout;
    const pad = 14;
    const rightW = Math.round(Math.max(232, Math.min(360, w * 0.3)));
    const rightX = w - rightW - pad;
    const labelW = 64;
    const gutterW = w < 1100 ? 84 : 104;
    const towerX0 = pad + labelW;
    const gutterX = rightX - pad - gutterW;
    const towerX1 = gutterX - 8;
    const L = this.cfg?.nLayers ?? 4;
    const tokH = 30;
    const outH = 30;
    const embH = Math.round(Math.max(60, Math.min(118, h * 0.14)));
    const avail = h - pad * 2 - tokH - outH - embH - 8;
    const per = avail / L;
    const rows: { y: number; h: number }[] = [];
    let y = h - pad - tokH;
    rows.push({ y, h: tokH });
    y -= embH + 4;
    rows.push({ y, h: embH });
    for (let l = 0; l < L; l++) {
      const ah = Math.round(per * 0.6), mh = Math.round(per * 0.4);
      y -= ah;
      rows.push({ y, h: ah });
      y -= mh;
      rows.push({ y, h: mh });
    }
    y -= outH;
    rows.push({ y, h: outH });
    const W = 16;
    const slots = Math.max(6, Math.min(W, Math.floor((towerX1 - towerX0) / 46)));
    const slotW = (towerX1 - towerX0) / slots;
    this.layout = { w, h, towerX0, towerX1, gutterX, rightX, rightW, rows, slots, slotW };
    return this.layout;
  }

  // ---------------------------------------------------------------------------------------------
  // frame

  /**
   * Rendering is split in two: the scene (cached in an offscreen canvas and only re-rendered while
   * something is tweening, about 1 s after each trace) and a cheap overlay drawn every frame
   * (flow particles, hover readouts). Idle frames cost one blit.
   */
  frame(now: number): void {
    const { w, h, dpr } = fitCanvas(this.canvas);
    const dt = this.lastFrame ? Math.min(50, now - this.lastFrame) : 16;
    this.lastFrame = now;
    this.flowT += dt;
    const prevLayout = this.layout;
    const lay = this.computeLayout();
    if (prevLayout !== lay) {
      if (this.trace) this.retarget(now, false);
      this.bump();
    }
    const vis = this.visCtx;
    if (this.cache.width !== this.canvas.width || this.cache.height !== this.canvas.height) {
      this.cache.width = this.canvas.width;
      this.cache.height = this.canvas.height;
      this.bump();
    }
    const dirty = now < this.animUntil || this.stage.active(now) || !this.trace;
    // While animating, draw straight onto the visible canvas. When things settle, render the scene
    // once more into the cache; idle frames then cost one blit plus the overlay.
    if (dirty || !this.cacheValid) {
      this.ctx = dirty ? vis : this.cacheCtx;
      const ctx = this.ctx;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      resetCtx(ctx);
      ctx.fillStyle = C.bg;
      ctx.fillRect(0, 0, w, h);
      if (!this.trace || !this.cfg) this.drawWaiting(w, h, now);
      else if (this.mode === "heads") this.drawHeads(now, w, h);
      else this.drawTower(now, lay);
      this.renders++;
      this.cacheValid = !dirty;
    }
    if (!dirty) {
      vis.setTransform(1, 0, 0, 1, 0, 0);
      vis.drawImage(this.cache, 0, 0);
    }
    if (!this.trace || !this.cfg) return;
    this.ctx = vis;
    vis.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.mode === "heads") this.drawHeadsHover(w);
    else {
      this.drawParticles(now, lay);
      this.drawHover(now, lay, [...this.items.values()]);
    }
  }

  /** Keep re-rendering the cached scene for `ms` (tweens are 300-650 ms). */
  bump(ms = 1100): void {
    this.animUntil = Math.max(this.animUntil, performance.now() + ms);
    this.cacheValid = false;
  }

  setMode(m: ViewMode): void {
    this.mode = m;
    this.bump();
  }

  setAllPairs(v: boolean): void {
    this.allPairs = v;
    this.bump();
  }

  private drawParticles(now: number, lay: Layout): void {
    const ctx = this.ctx;
    const tr = this.trace!;
    const S = tr.stages.length;
    const yBottom = lay.rows[1].y;
    const yTopAll = lay.rows[S].y;
    const fq = this.focusIdx();
    const s = this.stage.get(now);
    const reveal = s >= 98 ? 1 : clamp01((s - 1) / (lay.rows.length - 2));
    const yTop = yBottom + (yTopAll - yBottom) * reveal;
    const span = yBottom - yTopAll;
    for (const it of this.items.values()) {
      const a = it.alpha.get(now);
      if (a < 0.02) continue;
      const x = it.x.get(now);
      const isF = it.idx === fq && it.alive;
      ctx.fillStyle = withAlpha(isF ? C.yellow : C.blue, 0.6 * a);
      ctx.beginPath();
      for (let p = 0; p < 4; p++) {
        const f = (this.flowT / 2600 + p / 4 + (it.key % 7) * 0.13) % 1;
        const y = yBottom - f * span;
        if (y < yTop) continue;
        const xx = x + Math.sin(f * 9 + it.key) * 2;
        ctx.moveTo(xx + 1.3, y);
        ctx.arc(xx, y, 1.3, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }

  private drawWaiting(w: number, h: number, now: number): void {
    const ctx = this.ctx;
    ctx.fillStyle = C.dim;
    ctx.font = `15px ${FONT_SERIF}`;
    ctx.textAlign = "center";
    const dots = ".".repeat(1 + (Math.floor(now / 400) % 3));
    ctx.fillText(`waiting for the first forward pass${dots}`, w / 2, h / 2);
    ctx.textAlign = "left";
  }

  private rowAlpha(r: number, now: number): number {
    const s = this.stage.get(now);
    if (s >= 98) return 1;
    return 0.1 + 0.9 * clamp01(s - r + 1);
  }

  private rowDraw(r: number, now: number): number {
    const s = this.stage.get(now);
    if (s >= 98) return 1;
    return easeOutCubic(clamp01(s - r + 1));
  }

  private drawTower(now: number, lay: Layout): void {
    const ctx = this.ctx;
    const tr = this.trace!;
    const L = tr.nLayers, H = tr.nHeads, m = tr.dMlp;
    const nRows = lay.rows.length;
    const items = [...this.items.values()].sort((a, b) => a.key - b.key);
    const fq = this.focusIdx();
    const focusItem = items.find((it) => it.idx === fq && it.alive);

    // row bands + labels
    const pulseY = this.pulseY(now, lay);
    for (let r = 0; r < nRows; r++) {
      const row = lay.rows[r];
      const a = this.rowAlpha(r, now);
      const isAttn = r >= 2 && r < nRows - 1 && (r - 2) % 2 === 0;
      const l = r >= 2 && r < nRows - 1 ? Math.floor((r - 2) / 2) : -1;
      const dimLayer = this.selLayer >= 0 && l >= 0 && l !== this.selLayer;
      ctx.fillStyle = r % 2 === 0 ? "rgba(255,255,255,0.012)" : "rgba(255,255,255,0.028)";
      ctx.fillRect(lay.towerX0 - 60, row.y, lay.gutterX - lay.towerX0 + 60 + 96, row.h);
      if (pulseY !== null) {
        const dist = Math.abs(row.y + row.h / 2 - pulseY);
        const g = Math.exp(-(dist * dist) / (2 * 40 * 40));
        if (g > 0.02) {
          ctx.fillStyle = withAlpha(C.blue, 0.07 * g);
          ctx.fillRect(lay.towerX0 - 60, row.y, lay.gutterX - lay.towerX0 + 60 + 96, row.h);
        }
      }
      ctx.globalAlpha = a * (dimLayer ? 0.45 : 1);
      ctx.fillStyle = C.dim;
      ctx.font = `italic 13px ${FONT_SERIF}`;
      ctx.textBaseline = "middle";
      let label = "";
      if (r === 0) label = "tokens";
      else if (r === 1) label = "E[x]+P[t]";
      else if (r === nRows - 1) label = "unembed";
      else label = isAttn ? `attn ${l + 1}` : `mlp ${l + 1}`;
      ctx.fillText(label, 12, row.y + row.h / 2);
      if (r >= 2 && r < nRows - 1) {
        ctx.fillStyle = isAttn ? C.blue : C.yellow;
        ctx.fillRect(8, row.y + 6, 2, row.h - 12);
      }
      ctx.globalAlpha = 1;
    }

    // residual rivers (behind everything): width = measured ||x|| at each stage
    const nScale = this.normScale.get(now);
    const yAt = (s: number) => lay.rows[s + 1].y; // residual after stage s leaves the top of row s+1
    const yBottom = lay.rows[1].y;
    for (const it of items) {
      const a = it.alpha.get(now);
      if (a < 0.02) continue;
      const x = it.x.get(now);
      const nr = it.norms.get(now);
      const S = nr.length;
      const maxW = lay.slotW * 0.42;
      const isF = it === focusItem;
      const pts: [number, number][] = [];
      pts.push([yBottom, (nr[0] / nScale) * maxW]);
      for (let s = 0; s < S; s++) pts.push([yAt(s), (nr[s] / nScale) * maxW]);
      const reveal = this.stage.get(now) >= 98 ? 1 : clamp01((this.stage.get(now) - 1) / (nRows - 2));
      const yTop = yBottom + (yAt(S - 1) - yBottom) * reveal;
      ctx.beginPath();
      for (const [y, ww] of pts) if (y >= yTop - 0.5) ctx.lineTo(x - Math.max(1, ww / 2), y);
      for (let i = pts.length - 1; i >= 0; i--) { const [y, ww] = pts[i]; if (y >= yTop - 0.5) ctx.lineTo(x + Math.max(1, ww / 2), y); }
      ctx.closePath();
      const g = ctx.createLinearGradient(0, yBottom, 0, yAt(S - 1));
      const col = isF ? C.yellow : C.blue;
      g.addColorStop(0, withAlpha(col, 0.06 * a));
      g.addColorStop(1, withAlpha(col, (isF ? 0.24 : 0.13) * a));
      ctx.fillStyle = g;
      ctx.fill();
    }

    // attention arcs + MLP neuron grids per layer
    const mv = this.mlpVec.get(now);
    const ms = this.mlpScale.get(now);
    for (let l = 0; l < L; l++) {
      const ra = 2 + 2 * l, rm = 3 + 2 * l;
      const layerDim = this.selLayer >= 0 && l !== this.selLayer ? 0.28 : 1;
      this.drawArcs(now, lay, l, ra, items, focusItem, layerDim * this.rowAlpha(ra, now), this.rowDraw(ra, now));
      this.drawNeurons(now, lay, l, rm, mv.subarray(l * m, (l + 1) * m), ms, focusItem, layerDim * this.rowAlpha(rm, now));
    }
    void H;

    // embedding columns
    this.drawEmbeddings(now, lay, items, focusItem);
    // token chips
    this.drawChips(now, lay, items, focusItem);
    // logit lens gutter
    this.drawLens(now, lay);
    // output row + arrow into the right column
    this.drawOutputRow(now, lay, focusItem);
    // right column
    this.drawRight(now, lay, items, focusItem);
  }

  private pulseY(now: number, lay: Layout): number | null {
    const t = (now - this.pulseT0) / 700;
    if (t < 0 || t > 1.2) return null;
    const y0 = lay.rows[0].y + lay.rows[0].h, y1 = lay.rows[lay.rows.length - 1].y;
    return y0 + (y1 - y0) * easeInOutCubic(clamp01(t));
  }

  private drawArcs(now: number, lay: Layout, l: number, r: number, items: TokItem[], focus: TokItem | undefined, alpha: number, draw: number): void {
    const ctx = this.ctx;
    const tr = this.trace!;
    const H = tr.nHeads;
    const row = lay.rows[r];
    const yb = row.y + row.h - 3;
    const maxH = row.h - 8;
    ctx.save();
    ctx.beginPath();
    ctx.rect(lay.towerX0 - 20, row.y - 2, lay.towerX1 - lay.towerX0 + 40, row.h + 4);
    ctx.clip();
    ctx.lineCap = "round";
    // faint all-pairs arcs (every query inside the window), from the full matrices
    if (this.allPairs && draw > 0.98) {
      const w0 = tr.attention.windowStart, W = tr.attention.window;
      const byIdx = new Map<number, TokItem>();
      for (const it of items) if (it.alive) byIdx.set(it.idx, it);
      for (let h = 0; h < H; h++) {
        if (this.selHead >= 0 && h !== this.selHead) continue;
        const mat = tr.attention.matrices[l][h];
        const col = HEAD_COLORS[h % HEAD_COLORS.length];
        // two weight buckets, one path each (hundreds of arcs, two strokes)
        for (const [lo, hi, al, lw] of [[0.12, 0.35, 0.13, 0.8], [0.35, 1.01, 0.3, 1.6]] as const) {
          ctx.strokeStyle = withAlpha(col, al * alpha);
          ctx.lineWidth = lw;
          ctx.beginPath();
          for (const qi of byIdx.values()) {
            if (focus && qi === focus) continue;
            const qrow = qi.idx - w0;
            if (qrow < 0) continue;
            const xq = qi.x.get(now);
            for (const ki of byIdx.values()) {
              if (ki.idx >= qi.idx) continue;
              const wgt = mat[qrow * W + (ki.idx - w0)];
              if (!(wgt >= lo && wgt < hi)) continue;
              const xk = ki.x.get(now);
              const hgt = Math.min(maxH, 8 + 0.5 * Math.abs(xq - xk)) * (0.62 + 0.1 * h);
              ctx.moveTo(xq, yb);
              ctx.bezierCurveTo(xq, yb - hgt, xk, yb - hgt, xk, yb);
            }
          }
          ctx.stroke();
        }
      }
    }
    if (focus) {
      const xq = focus.x.get(now);
      for (let h = 0; h < H; h++) {
        if (this.selHead >= 0 && h !== this.selHead) continue;
        const col = HEAD_COLORS[h % HEAD_COLORS.length];
        for (const it of items) {
          const wgt = it.attn.get(now)[l * H + h] * it.alpha.get(now);
          if (wgt < 0.015) continue;
          const xk = it.x.get(now);
          ctx.strokeStyle = withAlpha(col, Math.min(1, 0.15 + wgt * 1.25) * alpha);
          ctx.lineWidth = 0.7 + 4.2 * wgt;
          ctx.beginPath();
          if (it === focus) {
            const rr = 5 + 2 * h;
            ctx.ellipse(xq, yb - rr, 4 + h, rr, 0, Math.PI * 0.5, Math.PI * 0.5 + Math.PI * 2 * draw);
          } else {
            const hgt = Math.min(maxH, 10 + 0.55 * Math.abs(xq - xk)) * (0.7 + 0.1 * h);
            bezierPartial(ctx, xq, yb, xq, yb - hgt, xk, yb - hgt, xk, yb, draw);
          }
          ctx.stroke();
          if (draw >= 1 && wgt > 0.25) {
            ctx.fillStyle = withAlpha(col, alpha);
            ctx.beginPath();
            ctx.arc(xk, yb, 1.6 + 2.2 * wgt, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
      ctx.fillStyle = withAlpha(C.yellow, alpha);
      ctx.beginPath();
      ctx.arc(xq, yb, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  private drawNeurons(now: number, lay: Layout, l: number, r: number, vals: Float32Array, scale: number, focus: TokItem | undefined, alpha: number): void {
    const ctx = this.ctx;
    const row = lay.rows[r];
    const m = vals.length;
    const cols = m >= 256 ? 64 : 32;
    const nrow = Math.ceil(m / cols);
    const x0 = lay.towerX0 + 6, x1 = lay.towerX1 - 6;
    const y0 = row.y + 6, y1 = row.y + row.h - 6;
    const dx = (x1 - x0) / (cols - 1), dy = nrow > 1 ? (y1 - y0) / (nrow - 1) : 0;
    ctx.globalAlpha = alpha;
    // resting neurons in one path
    ctx.fillStyle = "#2a3441";
    ctx.beginPath();
    for (let n = 0; n < m; n++) {
      const a = vals[n] / scale;
      if (a > 0.06) continue;
      const x = x0 + (n % cols) * dx, y = y0 + Math.floor(n / cols) * dy;
      ctx.rect(x - 0.9, y - 0.9, 1.8, 1.8);
    }
    ctx.fill();
    // lit neurons glow with the post-GELU activation
    const lit: { n: number; a: number }[] = [];
    for (let n = 0; n < m; n++) {
      const a = vals[n] / scale;
      if (a <= 0.06) continue;
      lit.push({ n, a });
      const x = x0 + (n % cols) * dx, y = y0 + Math.floor(n / cols) * dy;
      const rad = 1.2 + 3.4 * a;
      if (a > 0.3) {
        ctx.fillStyle = withAlpha(C.yellow, 0.13 * a);
        ctx.beginPath();
        ctx.arc(x, y, rad * 2.6, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = withAlpha(a > 0.55 ? "#FFF6C8" : C.yellow, 0.35 + 0.65 * a);
      ctx.beginPath();
      ctx.arc(x, y, rad, 0, Math.PI * 2);
      ctx.fill();
    }
    // the strongest neurons write back into the focus token's residual stream
    if (focus) {
      lit.sort((a, b) => b.a - a.a);
      const fx = focus.x.get(now);
      ctx.lineWidth = 1;
      for (const { n, a } of lit.slice(0, 3)) {
        const x = x0 + (n % cols) * dx, y = y0 + Math.floor(n / cols) * dy;
        ctx.strokeStyle = withAlpha(C.teal, 0.55 * a);
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.quadraticCurveTo((x + fx) / 2, row.y - 2, fx, row.y + 1);
        ctx.stroke();
        if (a > 0.5) {
          ctx.fillStyle = withAlpha(C.yellow, 0.9);
          ctx.font = `10px ${FONT_MONO}`;
          ctx.textBaseline = "bottom";
          ctx.fillText(`#${n}`, x + 4, y - 2);
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  private embCanvas = document.createElement("canvas");

  private drawEmbeddings(now: number, lay: Layout, items: TokItem[], focus: TokItem | undefined): void {
    const ctx = this.ctx;
    const row = lay.rows[1];
    const a0 = this.rowAlpha(1, now);
    const draw = this.rowDraw(1, now);
    const sc = this.embScale.get(now);
    const vis = items.filter((it) => it.alpha.get(now) * a0 >= 0.02);
    if (!vis.length) return;
    const d = vis[0].emb.length;
    const ec = 4, er = Math.ceil(d / ec);
    // one pixel per cell: block i holds token i, dimension j at (j % ec, er - 1 - floor(j / ec))
    const ec2 = this.embCanvas;
    if (ec2.width !== ec * vis.length || ec2.height !== er) { ec2.width = ec * vis.length; ec2.height = er; }
    const ectx = ec2.getContext("2d")!;
    const img = ectx.createImageData(ec2.width, er);
    vis.forEach((it, i) => {
      const e = it.emb.get(now);
      for (let j = 0; j < d; j++) {
        const k = divIndex(e[j], sc);
        const o = ((er - 1 - Math.floor(j / ec)) * ec2.width + i * ec + (j % ec)) * 4;
        img.data[o] = DIV_LUT[k * 3]; img.data[o + 1] = DIV_LUT[k * 3 + 1]; img.data[o + 2] = DIV_LUT[k * 3 + 2]; img.data[o + 3] = 255;
      }
    });
    ectx.putImageData(img, 0, 0);
    const cw = Math.max(2, Math.min(9, (lay.slotW * 0.62) / ec));
    const ch = (row.h - 8) / er;
    const shown = Math.max(0, Math.min(er, Math.floor(er * draw)));
    const yb = row.y + row.h - 4;
    ctx.imageSmoothingEnabled = false;
    ctx.strokeStyle = C.bg;
    ctx.lineWidth = 0.7;
    vis.forEach((it, i) => {
      if (!shown) return;
      const a = it.alpha.get(now) * a0;
      const x = it.x.get(now) - (cw * ec) / 2;
      ctx.globalAlpha = a;
      ctx.drawImage(ec2, i * ec, er - shown, ec, shown, x, yb - shown * ch, cw * ec, shown * ch);
      ctx.beginPath();
      for (let c = 1; c < ec; c++) { ctx.moveTo(x + c * cw, yb - shown * ch); ctx.lineTo(x + c * cw, yb); }
      if (ch >= 2.4) for (let r = 1; r < shown; r++) { ctx.moveTo(x, yb - r * ch); ctx.lineTo(x + cw * ec, yb - r * ch); }
      ctx.stroke();
      if (it === focus) {
        ctx.strokeStyle = withAlpha(C.yellow, 0.8);
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 2, row.y + 1, cw * ec + 3, row.h - 4);
        ctx.strokeStyle = C.bg;
        ctx.lineWidth = 0.7;
      }
    });
    ctx.globalAlpha = 1;
  }

  private drawChips(now: number, lay: Layout, items: TokItem[], focus: TokItem | undefined): void {
    const ctx = this.ctx;
    const row = lay.rows[0];
    ctx.font = `11.5px ${FONT_MONO}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    for (const it of items) {
      const a = it.alpha.get(now);
      if (a < 0.02) continue;
      const x = it.x.get(now);
      const label = showToken(it.text);
      const maxW = lay.slotW - 5;
      let txt = label;
      while (ctx.measureText(txt).width > maxW - 8 && txt.length > 1) txt = txt.slice(0, -1);
      if (txt !== label) txt = txt.slice(0, -1) + "…";
      const cw = Math.min(maxW, ctx.measureText(txt).width + 12);
      const isF = it === focus, isH = it.key === this.hoverKey;
      ctx.globalAlpha = a;
      roundRect(ctx, x - cw / 2, row.y + 4, cw, row.h - 8, 5);
      ctx.fillStyle = isF ? "rgba(247,217,111,0.14)" : C.panel2;
      ctx.fill();
      ctx.strokeStyle = isF ? C.yellow : isH ? C.blue : C.line;
      ctx.lineWidth = isF || isH ? 1.4 : 1;
      ctx.stroke();
      ctx.fillStyle = isF ? C.yellow : it.text.startsWith("<|") ? C.dim : C.text;
      ctx.fillText(txt, x, row.y + row.h / 2 + 0.5);
      ctx.globalAlpha = 1;
    }
    ctx.textAlign = "left";
  }

  private drawLens(now: number, lay: Layout): void {
    const ctx = this.ctx;
    const tr = this.trace!;
    const following = this.focusIdx() === tr.T - 1;
    const x = lay.gutterX + 4;
    const gw = lay.rightX - 14 - x;
    ctx.font = `10px ${FONT_SANS}`;
    ctx.fillStyle = C.faint;
    ctx.textBaseline = "alphabetic";
    ctx.fillText("↑ logit lens", x, lay.rows[0].y + lay.rows[0].h / 2 + 4);
    for (let s = 0; s < this.lens.length; s++) {
      const r = s + 1;
      const row = lay.rows[r];
      const ln = this.lens[s];
      const a = this.rowAlpha(r, now) * (following ? 1 : 0.35);
      const flash = clamp01(1 - (now - ln.t0) / 900);
      const y = row.y + 2;
      // the guess sits at the top edge of the row: the residual after this stage
      ctx.globalAlpha = a;
      if (flash > 0) {
        ctx.fillStyle = withAlpha(C.yellow, 0.18 * flash);
        roundRect(ctx, x - 2, y, gw, 18, 4);
        ctx.fill();
      }
      ctx.font = `11.5px ${FONT_MONO}`;
      ctx.textBaseline = "middle";
      ctx.fillStyle = s === this.lens.length - 1 ? C.yellow : C.text;
      let t = "→ " + showToken(ln.text);
      while (ctx.measureText(t).width > gw - 34 && t.length > 3) t = t.slice(0, -1);
      ctx.fillText(t, x, y + 9);
      ctx.font = `10px ${FONT_SANS}`;
      ctx.fillStyle = C.dim;
      ctx.textAlign = "right";
      ctx.fillText(pct(ln.p, 0), x + gw - 4, y + 9);
      ctx.textAlign = "left";
      ctx.fillStyle = withAlpha(C.blue, 0.7);
      ctx.fillRect(x, y + 16, (gw - 6) * ln.p, 2);
      ctx.globalAlpha = 1;
    }
  }

  private drawOutputRow(now: number, lay: Layout, focus: TokItem | undefined): void {
    const ctx = this.ctx;
    const r = lay.rows.length - 1;
    const row = lay.rows[r];
    const a = this.rowAlpha(r, now);
    ctx.globalAlpha = a;
    ctx.font = `italic 14px ${FONT_SERIF}`;
    ctx.fillStyle = C.text;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    const midX = (lay.towerX0 + lay.towerX1) / 2;
    ctx.fillText("p(xₜ₊₁ | x≤ₜ) = softmax( LN(x) · Eᵀ )", midX, row.y + row.h / 2 - 2);
    ctx.textAlign = "left";
    // arrow from the last token's stream into the softmax panel
    const last = [...this.items.values()].filter((i) => i.alive).sort((x, y) => y.key - x.key)[0];
    if (last) {
      const t = clamp01((now - this.pulseT0 - 450) / 450) * this.rowDraw(r, now);
      const x0 = last.x.get(now), y0 = lay.rows[r - 1].y + 2;
      const x3 = lay.rightX - 6, y3 = 44;
      ctx.strokeStyle = withAlpha(C.yellow, 0.75);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      bezierPartial(ctx, x0, y0, x0, y0 - 40, x3 - 60, y3, x3, y3, easeInOutCubic(t));
      ctx.stroke();
      if (t >= 1) {
        ctx.fillStyle = C.yellow;
        ctx.beginPath();
        ctx.moveTo(x3 + 2, y3);
        ctx.lineTo(x3 - 7, y3 - 4.5);
        ctx.lineTo(x3 - 7, y3 + 4.5);
        ctx.closePath();
        ctx.fill();
      }
    }
    void focus;
    ctx.globalAlpha = 1;
  }

  private drawRight(now: number, lay: Layout, items: TokItem[], focus: TokItem | undefined): void {
    const ctx = this.ctx;
    const x = lay.rightX, w = lay.rightW, h = lay.h;
    const tr = this.trace!;
    const pad = 12;
    // panel frames
    const barsH = Math.round(Math.min(330, h * 0.4));
    const pcaH = Math.round(Math.min(280, (h - barsH) * 0.55));
    const stepsY = 14 + barsH + 10 + pcaH + 10;
    panel(ctx, x, 14, w, barsH);
    panel(ctx, x, 14 + barsH + 10, w, pcaH);
    panel(ctx, x, stepsY, w, h - stepsY - 14);

    // softmax bars
    ctx.font = `600 12px ${FONT_SANS}`;
    ctx.fillStyle = C.text;
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Next token", x + pad, 14 + 22);
    ctx.font = `italic 12px ${FONT_SERIF}`;
    ctx.fillStyle = C.dim;
    ctx.textAlign = "right";
    ctx.fillText(`H = ${fmt(tr.next.entropy, 2)} nats`, x + w - pad, 14 + 22);
    ctx.textAlign = "left";
    const top = 14 + 34, rowH = Math.min(26, (barsH - 46) / 10);
    const labW = Math.min(110, w * 0.34);
    const maxP = Math.max(0.25, tr.next.topk[0]?.p ?? 0.25);
    const bars = [...this.bars.values()];
    for (const b of bars) {
      const a = b.alpha.get(now);
      if (a < 0.02) continue;
      const rk = b.rank.get(now);
      const y = top + rk * rowH;
      const p = b.p.get(now);
      const bw = (w - pad * 2 - labW - 44) * Math.min(1, p / maxP);
      ctx.globalAlpha = a;
      ctx.font = `12px ${FONT_MONO}`;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillStyle = rk < 0.5 ? C.yellow : C.text;
      let t = showToken(b.text);
      while (ctx.measureText(t).width > labW - 8 && t.length > 1) t = t.slice(0, -1);
      ctx.fillText(t, x + pad + labW - 6, y + rowH / 2);
      ctx.textAlign = "left";
      const g = ctx.createLinearGradient(x + pad + labW, 0, x + pad + labW + bw + 1, 0);
      const col = rk < 0.5 ? C.yellow : C.blue;
      g.addColorStop(0, withAlpha(col, 0.35));
      g.addColorStop(1, withAlpha(col, 0.95));
      ctx.fillStyle = g;
      roundRect(ctx, x + pad + labW, y + 3, Math.max(1, bw), rowH - 6, 3);
      ctx.fill();
      ctx.font = `11px ${FONT_SANS}`;
      ctx.fillStyle = C.dim;
      ctx.fillText(pct(p, p < 0.1 ? 1 : 0), x + pad + labW + bw + 6, y + rowH / 2);
      ctx.globalAlpha = 1;
    }

    // residual stream PCA: tokens at the final stage + the focus token's path through the layers
    const py = 14 + barsH + 10;
    ctx.font = `600 12px ${FONT_SANS}`;
    ctx.fillStyle = C.text;
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Residual stream", x + pad, py + 22);
    ctx.font = `italic 12px ${FONT_SERIF}`;
    ctx.fillStyle = C.dim;
    ctx.textAlign = "right";
    ctx.fillText("PC1 × PC2, fixed basis", x + w - pad, py + 22);
    ctx.textAlign = "left";
    const cx = x + w / 2, cy = py + 30 + (pcaH - 40) / 2;
    const R = Math.min(w - 2 * pad, pcaH - 44) / 2;
    const sc = R / (this.pcaScale.get(now) * 1.6);
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
    ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R);
    ctx.stroke();
    const S = tr.stages.length;
    const stageNow = this.stage.get(now) >= 98 ? S - 1 : Math.max(0, Math.min(S - 1, Math.floor(this.stage.get(now)) - 1));
    for (const it of items) {
      const a = it.alpha.get(now);
      if (a < 0.02 || it === focus) continue;
      const p = it.pca.get(now);
      ctx.fillStyle = withAlpha(C.blue, 0.6 * a);
      ctx.beginPath();
      const [qx, qy] = clampR(p[stageNow * 3] * sc, p[stageNow * 3 + 1] * sc, R);
      ctx.arc(cx + qx, cy - qy, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
    if (focus) {
      const p = focus.pca.get(now);
      const t = clamp01((now - this.pulseT0) / 900);
      const segs = Math.min(stageNow, S - 1);
      for (let s = 0; s < segs; s++) {
        const f = clamp01(t * segs - s);
        if (f <= 0) break;
        const [ux, uy] = clampR(p[s * 3] * sc, p[s * 3 + 1] * sc, R);
        const [vx, vy] = clampR(p[(s + 1) * 3] * sc, p[(s + 1) * 3 + 1] * sc, R);
        const ax = cx + ux, ay = cy - uy, bx = cx + vx, by = cy - vy;
        arrow(ctx, ax, ay, ax + (bx - ax) * f, ay + (by - ay) * f, s % 2 === 0 ? C.blue : C.yellow, f >= 1);
      }
      const [gx, gy] = clampR(p[stageNow * 3] * sc, p[stageNow * 3 + 1] * sc, R);
      const fx = cx + gx, fy = cy - gy;
      ctx.fillStyle = C.yellow;
      ctx.beginPath();
      ctx.arc(fx, fy, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = `11px ${FONT_MONO}`;
      ctx.fillText(showToken(focus.text).slice(0, 14), fx + 7, fy - 6);
      ctx.fillStyle = C.faint;
      ctx.font = `10px ${FONT_SANS}`;
      const [ex, ey] = clampR(p[0] * sc, p[1] * sc, R);
      ctx.fillText("embed", cx + ex + 5, cy - ey + 11);
    }

    // greedy completion steps from POST /api/complete
    ctx.font = `600 12px ${FONT_SANS}`;
    ctx.fillStyle = C.text;
    ctx.fillText("Greedy completion", x + pad, stepsY + 22);
    ctx.font = `11px ${FONT_SANS}`;
    ctx.fillStyle = C.dim;
    ctx.textAlign = "right";
    if (this.completion) ctx.fillText(`${this.completion.steps.length} tok · ${this.completion.ms} ms`, x + w - pad, stepsY + 22);
    ctx.textAlign = "left";
    const avail = h - stepsY - 14 - 36;
    const sh = Math.max(16, Math.min(22, avail / Math.max(1, this.steps.length)));
    let yy = stepsY + 34;
    for (const { s, at } of this.steps) {
      const a = clamp01((now - at) / 250);
      if (a <= 0 || yy + sh > h - 18) break;
      ctx.globalAlpha = a;
      ctx.font = `12px ${FONT_MONO}`;
      ctx.fillStyle = C.yellow;
      ctx.textBaseline = "middle";
      ctx.fillText(showToken(s.text).slice(0, 14), x + pad, yy + sh / 2);
      const bx = x + pad + Math.min(118, w * 0.38), bw = w - pad - (bx - x) - 64;
      ctx.fillStyle = C.grid;
      ctx.fillRect(bx, yy + sh / 2 - 3, bw, 6);
      ctx.fillStyle = withAlpha(C.teal, 0.85);
      ctx.fillRect(bx, yy + sh / 2 - 3, bw * s.p * easeOutCubic(a), 6);
      ctx.font = `10.5px ${FONT_SANS}`;
      ctx.fillStyle = C.dim;
      ctx.fillText(`${pct(s.p, 0)}  H ${fmt(s.entropy, 1)}`, bx + bw + 6, yy + sh / 2);
      ctx.globalAlpha = 1;
      yy += sh;
    }
    if (!this.steps.length) {
      ctx.font = `italic 12px ${FONT_SERIF}`;
      ctx.fillStyle = C.faint;
      ctx.fillText("waiting for /api/complete…", x + pad, stepsY + 46);
    }
  }

  private drawHover(now: number, lay: Layout, items: TokItem[]): void {
    const ctx = this.ctx;
    const ms = this.mouse;
    let hk: number | null = null;
    if (ms) {
      const row0 = lay.rows[0];
      if (ms.y >= lay.rows[1].y && ms.y <= row0.y + row0.h) {
        for (const it of items) if (it.alive && Math.abs(it.x.get(now) - ms.x) < lay.slotW / 2) hk = it.key;
      }
      // neuron hover
      const L = this.trace!.nLayers, m = this.trace!.dMlp;
      for (let l = 0; l < L; l++) {
        const row = lay.rows[3 + 2 * l];
        if (ms.y < row.y || ms.y > row.y + row.h || ms.x < lay.towerX0 || ms.x > lay.towerX1) continue;
        const cols = m >= 256 ? 64 : 32, nrow = Math.ceil(m / cols);
        const x0 = lay.towerX0 + 6, x1 = lay.towerX1 - 6, y0 = row.y + 6, y1 = row.y + row.h - 6;
        const c = Math.round(((ms.x - x0) / (x1 - x0)) * (cols - 1));
        const r = Math.round(((ms.y - y0) / Math.max(1, y1 - y0)) * (nrow - 1));
        const n = r * cols + c;
        if (c >= 0 && c < cols && r >= 0 && r < nrow && n < m) {
          const v = this.mlpVec.target[l * m + n];
          tooltip(ctx, ms.x, ms.y, [`layer ${l + 1} · neuron #${n}`, `GELU activation ${fmt(v, 3)}`], lay.w);
        }
      }
    }
    if (hk !== this.hoverKey) {
      this.hoverKey = hk;
      this.bump(50);
      this.cb.onHoverToken(hk === null ? null : this.charRange(hk));
    }
    if (hk !== null && ms) {
      const it = this.items.get(hk)!;
      const n = it.norms.target;
      tooltip(ctx, ms.x, ms.y, [`token #${it.id}  '${showToken(it.text)}'`, `position ${it.idx}`, `‖x‖: embed ${fmt(n[0], 2)} → out ${fmt(n[n.length - 1], 2)}`, "click to make it the query"], lay.w);
    }
  }

  private charRange(key: number): { from: number; to: number } | null {
    const tk = this.tokens;
    if (!tk) return null;
    const idx = key - tk.truncated;
    let end = tk.prefixChars;
    for (let i = tk.tokens.length - 1; i > idx; i--) end -= tk.tokens[i].text.length;
    const t = tk.tokens[idx];
    if (!t || t.text.startsWith("<|")) return null;
    return { from: Math.max(0, end - t.text.length), to: Math.max(0, end) };
  }

  private onClick(): void {
    if (this.mode === "heads") {
      if (this.hoverHeadQuery !== null) {
        const tk = this.tokens!;
        const key = tk.truncated + this.hoverHeadQuery;
        this.setFocus(this.hoverHeadQuery === this.trace!.T - 1 ? null : key);
      }
      return;
    }
    if (this.hoverKey !== null) {
      const last = Math.max(...this.order);
      this.setFocus(this.hoverKey === last || this.hoverKey === this.focusKey ? null : this.hoverKey);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // heads close-up

  private hoverHeadQuery: number | null = null;
  private headGeom: { hh: number; mx0: number; my0: number; cell: number }[] = [];

  private drawHeads(now: number, w: number, h: number): void {
    const ctx = this.ctx;
    const tr = this.trace!, tk = this.tokens!;
    const H = tr.nHeads, W = tr.attention.window, w0 = tr.attention.windowStart;
    const l = this.selLayer >= 0 ? this.selLayer : tr.nLayers - 1;
    const mats = this.headMats.get(now);
    const fq = this.focusIdx();
    const qi = fq - w0;
    ctx.font = `600 15px ${FONT_SANS}`;
    ctx.fillStyle = C.text;
    ctx.textBaseline = "alphabetic";
    ctx.fillText(`Layer ${l + 1} attention, close-up`, 18, 30);
    const tw = ctx.measureText(`Layer ${l + 1} attention, close-up`).width;
    ctx.font = `italic 15px ${FONT_SERIF}`;
    ctx.fillStyle = C.dim;
    ctx.fillText(`A = softmax( QKᵀ / √${tr.dModel / H} + causal mask ),  last ${W} of ${tr.T} tokens`, 18 + tw + 24, 30);
    const cols = Math.ceil(Math.sqrt(H)), rowsN = Math.ceil(H / cols);
    const gap = 12;
    const gx = 14, gy = 44, gw = (w - gx * 2 - (cols - 1) * gap) / cols, gh = (h - gy - 12 - (rowsN - 1) * gap) / rowsN;
    this.headGeom = [];
    for (let hh = 0; hh < H; hh++) {
      const px = gx + (hh % cols) * (gw + gap), py = gy + Math.floor(hh / cols) * (gh + gap);
      const col = HEAD_COLORS[hh % HEAD_COLORS.length];
      const dim = this.selHead >= 0 && hh !== this.selHead ? 0.35 : 1;
      ctx.globalAlpha = dim;
      panel(ctx, px, py, gw, gh);
      const M = mats.subarray(hh * W * W, (hh + 1) * W * W);
      // head statistics measured on this trace (queries inside the window)
      let prev = 0, dist = 0, ent = 0, mx = 1e-6;
      for (let q = 0; q < W; q++) {
        let rs = 0;
        for (let k = 0; k <= q; k++) {
          const v = M[q * W + k];
          if (k === q - 1) prev += v;
          dist += v * (q - k);
          if (v > 0) rs -= v * Math.log(v);
          if (v > mx) mx = v;
        }
        ent += rs;
      }
      const bof = tr.attention.last[l][hh][0] ?? 0;
      ctx.fillStyle = col;
      ctx.font = `600 13px ${FONT_SANS}`;
      ctx.fillText(`head ${hh + 1}`, px + 12, py + 20);
      ctx.font = `11px ${FONT_SANS}`;
      ctx.fillStyle = C.dim;
      ctx.fillText(`prev-token ${fmt(prev / Math.max(1, W - 1), 2)} · distance ${fmt(dist / W, 1)} · entropy ${fmt(ent / W, 2)} nats · last query → ⟨bof⟩ ${pct(bof, 0)}`, px + 70, py + 20);
      // matrix (left) + token column with arcs (right), rows aligned
      const top = py + 34, bodyH = gh - 34 - 22;
      const labW = Math.min(78, gw * 0.14);
      const arcW = Math.min(230, gw * 0.34);
      const cell = Math.max(4, Math.min((gw - labW - arcW - 34) / W, bodyH / W));
      const mx0 = px + 12 + labW, my0 = top + (bodyH - cell * W) / 2;
      this.headGeom.push({ hh, mx0, my0, cell });
      for (let q = 0; q < W; q++) {
        for (let k = 0; k < W; k++) {
          const x0 = mx0 + k * cell, y0 = my0 + q * cell;
          const v = M[q * W + k];
          ctx.fillStyle = k > q ? "#0f1318" : seqColor(Math.sqrt(v / mx));
          ctx.fillRect(x0, y0, cell - 1, cell - 1);
          if (cell > 24 && k <= q && v >= 0.05) {
            ctx.fillStyle = v / mx > 0.45 ? "#0e1116" : C.text;
            ctx.font = `9.5px ${FONT_SANS}`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(v.toFixed(2).replace(/^0/, ""), x0 + cell / 2, y0 + cell / 2);
            ctx.textAlign = "left";
          }
        }
      }
      // query labels left of the matrix
      ctx.font = `10.5px ${FONT_MONO}`;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let q = 0; q < W; q++) {
        const t = tk.tokens[w0 + q];
        ctx.fillStyle = q === qi ? C.yellow : C.dim;
        ctx.fillText(clip(ctx, showToken(t?.text ?? ""), labW - 6), mx0 - 6, my0 + q * cell + cell / 2);
      }
      ctx.textAlign = "left";
      if (qi >= 0) {
        ctx.strokeStyle = C.yellow;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(mx0 - 1, my0 + qi * cell - 1, cell * W + 1, cell + 1);
      }
      // arc column: the same tokens top to bottom; arcs bulge right from the query to each key
      const tx = mx0 + W * cell + 18;
      const ax = tx + Math.min(70, arcW * 0.36);
      ctx.font = `10.5px ${FONT_MONO}`;
      for (let k = 0; k < W; k++) {
        const t = tk.tokens[w0 + k];
        ctx.fillStyle = k === qi ? C.yellow : C.dim;
        ctx.fillText(clip(ctx, showToken(t?.text ?? ""), ax - tx - 8), tx, my0 + k * cell + cell / 2);
      }
      ctx.lineCap = "round";
      const yOf = (i: number) => my0 + i * cell + cell / 2;
      const bulge = (dq: number) => Math.min(arcW - (ax - tx) - 10, 10 + dq * cell * 0.55);
      if (this.allPairs) {
        ctx.strokeStyle = withAlpha(col, 0.13);
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        for (let q = 0; q < W; q++) {
          if (q === qi) continue;
          for (let k = 0; k < q; k++) {
            if (M[q * W + k] < 0.12) continue;
            const b = bulge(q - k);
            ctx.moveTo(ax, yOf(q));
            ctx.bezierCurveTo(ax + b, yOf(q), ax + b, yOf(k), ax, yOf(k));
          }
        }
        ctx.stroke();
      }
      if (qi >= 0) {
        let rowMax = 1e-6;
        for (let k = 0; k <= qi; k++) rowMax = Math.max(rowMax, M[qi * W + k]);
        const draw = easeOutCubic(clamp01((now - this.pulseT0) / 700));
        for (let k = 0; k <= qi; k++) {
          const v = M[qi * W + k];
          if (v < 0.01) continue;
          const rel = v / rowMax;
          ctx.strokeStyle = withAlpha(col, 0.15 + 0.85 * rel);
          ctx.lineWidth = 0.8 + 5 * v;
          ctx.beginPath();
          if (k === qi) ctx.ellipse(ax + 7, yOf(qi), 7, Math.max(3, cell * 0.32), 0, Math.PI, Math.PI + Math.PI * 2 * draw);
          else {
            const b = bulge(qi - k);
            bezierPartial(ctx, ax, yOf(qi), ax + b, yOf(qi), ax + b, yOf(k), ax, yOf(k), draw);
          }
          ctx.stroke();
          ctx.fillStyle = withAlpha(col, 0.3 + 0.7 * rel);
          ctx.beginPath();
          ctx.arc(ax, yOf(k), 1.5 + 2.5 * rel, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = C.yellow;
        ctx.beginPath();
        ctx.arc(ax, yOf(qi), 3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.font = `italic 11px ${FONT_SERIF}`;
      ctx.fillStyle = C.faint;
      ctx.textBaseline = "alphabetic";
      ctx.fillText(`rows: query · columns: key · colour √(a / ${fmt(mx, 2)}) · arcs: query row, opacity ∝ a / row max`, px + 12, py + gh - 8);
      ctx.globalAlpha = 1;
    }
  }

  private drawHeadsHover(w: number): void {
    const ms = this.mouse;
    this.hoverHeadQuery = null;
    if (!ms || !this.trace || !this.tokens) return;
    const tr = this.trace, tk = this.tokens;
    const W = tr.attention.window, w0 = tr.attention.windowStart;
    const l = this.selLayer >= 0 ? this.selLayer : tr.nLayers - 1;
    for (const g of this.headGeom) {
      if (ms.x < g.mx0 || ms.x >= g.mx0 + g.cell * W || ms.y < g.my0 || ms.y >= g.my0 + g.cell * W) continue;
      const k = Math.floor((ms.x - g.mx0) / g.cell), q = Math.floor((ms.y - g.my0) / g.cell);
      this.hoverHeadQuery = w0 + q;
      const ctx = this.ctx;
      ctx.strokeStyle = C.text;
      ctx.lineWidth = 1;
      ctx.strokeRect(g.mx0 + k * g.cell - 0.5, g.my0 + q * g.cell - 0.5, g.cell, g.cell);
      tooltip(ctx, ms.x, ms.y, [
        `layer ${l + 1} head ${g.hh + 1}: '${showToken(tk.tokens[w0 + q]?.text ?? "")}' → '${showToken(tk.tokens[w0 + k]?.text ?? "")}'`,
        k > q ? "masked (future token)" : `attention ${fmt(tr.attention.matrices[l][g.hh][q * W + k], 3)}`,
        "click to make this row the query",
      ], w);
    }
  }
}

// -------------------------------------------------------------------------------------------------
// drawing helpers

/** Canvas state leaks between frames (e.g. a tooltip's textBaseline); start every render clean. */
export function resetCtx(ctx: CanvasRenderingContext2D): void {
  ctx.globalAlpha = 1;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  ctx.setLineDash([]);
  ctx.lineCap = "butt";
  ctx.imageSmoothingEnabled = true;
}

/** Keep a point inside the plot circle (outliers sit on the rim). */
function clampR(x: number, y: number, R: number): [number, number] {
  const r = Math.hypot(x, y);
  return r > R ? [(x / r) * R, (y / r) * R] : [x, y];
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function clip(ctx: CanvasRenderingContext2D, s: string, maxW: number): string {
  if (ctx.measureText(s).width <= maxW) return s;
  let t = s;
  while (t.length > 1 && ctx.measureText(t + "…").width > maxW) t = t.slice(0, -1);
  return t + "…";
}

export function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function panel(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
  roundRect(ctx, x, y, w, h, 10);
  ctx.fillStyle = C.panel;
  ctx.fill();
  ctx.strokeStyle = "#1e2530";
  ctx.lineWidth = 1;
  ctx.stroke();
}

function arrow(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, col: string, head: boolean): void {
  ctx.strokeStyle = withAlpha(col, 0.85);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  if (!head) return;
  const a = Math.atan2(y1 - y0, x1 - x0);
  const len = Math.hypot(x1 - x0, y1 - y0);
  if (len < 6) return;
  ctx.fillStyle = withAlpha(col, 0.9);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - 7 * Math.cos(a - 0.4), y1 - 7 * Math.sin(a - 0.4));
  ctx.lineTo(x1 - 7 * Math.cos(a + 0.4), y1 - 7 * Math.sin(a + 0.4));
  ctx.closePath();
  ctx.fill();
}

export function tooltip(ctx: CanvasRenderingContext2D, x: number, y: number, lines: string[], maxX: number): void {
  ctx.font = `11.5px ${FONT_SANS}`;
  const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 18;
  const h = lines.length * 16 + 10;
  let tx = x + 14, ty = y + 14;
  if (tx + w > maxX - 6) tx = x - w - 14;
  if (ty < 6) ty = 6;
  roundRect(ctx, tx, ty, w, h, 6);
  ctx.fillStyle = "rgba(14,17,22,0.95)";
  ctx.fill();
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.textBaseline = "top";
  lines.forEach((l, i) => {
    ctx.fillStyle = i === 0 ? C.text : C.dim;
    ctx.fillText(l, tx + 9, ty + 6 + i * 16);
  });
}
