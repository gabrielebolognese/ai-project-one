/**
 * Weights browser: every tensor of the PC-resident model (GET /api/model + raw Float32 from
 * /api/weights/:name?format=bin). Zoomable heatmap (wheel = zoom at cursor, drag = pan, double
 * click = reset) with a diverging colormap, exact hover readout W[i,j], the tensor's histogram and
 * per-layer Frobenius norms. Lenses: position-embedding cosine similarity and a 2-D PCA of the
 * vocabulary embedding (power iteration, computed in the browser).
 */
import { getTensor, type ModelInfo, type TensorStats } from "../api";
import { C, DIV_LUT, FONT_MONO, FONT_SANS, FONT_SERIF, HEAD_COLORS, SEQ_LUT, divColor, fmt, fmtInt, pct, showToken, withAlpha } from "../theme";
import { TweenNum, clamp01, easeOutCubic, fitCanvas } from "../tween";
import { resetCtx, roundRect, tooltip } from "./network";

type Lens = "heatmap" | "cosine" | "pca";

interface Loaded {
  name: string;
  shape: number[];
  data: Float32Array;
  rows: number;
  cols: number;
  transposed: boolean;
  scale: number;
  image: HTMLCanvasElement;
  rowNorms: Float32Array;
}

const JAVA_KEYWORDS = new Set(["public", "private", "protected", "class", "static", "final", "void", "int", "return", "new", "if", "else", "for", "while", "import", "package", "this", "null", "true", "false", "String", "throws", "throw", "try", "catch", "boolean", "long", "double", "extends", "implements", "interface", "Override", "List", "Map"]);

export class WeightsView {
  private listEl: HTMLElement;
  private mainCanvas: HTMLCanvasElement;
  private sideCanvas: HTMLCanvasElement;
  private headerEl: HTMLElement;
  private lensEl: HTMLElement;
  private info: ModelInfo | null = null;
  private cache = new Map<string, Loaded>();
  private cur: Loaded | null = null;
  private selected = "";
  private lens: Lens = "heatmap";
  private view = { zoom: 1, x: 0, y: 0 };
  private fit = { zoom: 1, x: 0, y: 0 };
  private appear = new TweenNum(0, 700);
  private mouse: { x: number; y: number } | null = null;
  private drag: { x: number; y: number; vx: number; vy: number } | null = null;
  private vocabTexts: string[] = [];
  private pcaPts: Float32Array | null = null;
  private cosine: HTMLCanvasElement | null = null;
  private loading = false;

  constructor(root: HTMLElement) {
    root.innerHTML = `
      <aside class="w-list"></aside>
      <section class="w-main">
        <div class="w-header"></div>
        <div class="w-lens"></div>
        <canvas class="w-canvas"></canvas>
      </section>
      <aside class="w-side"><canvas class="w-side-canvas"></canvas></aside>`;
    this.listEl = root.querySelector(".w-list")!;
    this.headerEl = root.querySelector(".w-header")!;
    this.lensEl = root.querySelector(".w-lens")!;
    this.mainCanvas = root.querySelector(".w-canvas")!;
    this.sideCanvas = root.querySelector(".w-side-canvas")!;
    const c = this.mainCanvas;
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const f = Math.exp(-e.deltaY * 0.0015);
      const z = Math.max(this.fit.zoom * 0.5, Math.min(this.fit.zoom * 400, this.view.zoom * f));
      const k = z / this.view.zoom;
      this.view.x = mx - (mx - this.view.x) * k;
      this.view.y = my - (my - this.view.y) * k;
      this.view.zoom = z;
    }, { passive: false });
    c.addEventListener("mousedown", (e) => { this.drag = { x: e.clientX, y: e.clientY, vx: this.view.x, vy: this.view.y }; });
    window.addEventListener("mouseup", () => { this.drag = null; });
    c.addEventListener("mousemove", (e) => {
      const r = c.getBoundingClientRect();
      this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
      if (this.drag) {
        this.view.x = this.drag.vx + (e.clientX - this.drag.x);
        this.view.y = this.drag.vy + (e.clientY - this.drag.y);
      }
    });
    c.addEventListener("mouseleave", () => { this.mouse = null; });
    c.addEventListener("dblclick", () => { this.view = { ...this.fit }; });
  }

  setModel(info: ModelInfo, vocabTexts: string[]): void {
    this.info = info;
    this.vocabTexts = vocabTexts;
    this.renderList();
    if (!this.selected) void this.select("tok_emb");
  }

  /** Zoom into a region of the current tensor (used by the screenshot script and keyboard). */
  zoomTo(factor: number): void {
    const r = this.mainCanvas.getBoundingClientRect();
    const mx = r.width / 2, my = r.height / 2;
    const z = this.fit.zoom * factor;
    const k = z / this.view.zoom;
    this.view.x = mx - (mx - this.view.x) * k;
    this.view.y = my - (my - this.view.y) * k;
    this.view.zoom = z;
  }

  private groups(): { title: string; items: TensorStats[] }[] {
    const t = this.info!.tensors;
    const L = this.info!.config.nLayers;
    const out: { title: string; items: TensorStats[] }[] = [{ title: "Embeddings", items: t.filter((x) => x.name === "tok_emb" || x.name === "pos_emb") }];
    for (let l = 0; l < L; l++) out.push({ title: `Block ${l + 1}`, items: t.filter((x) => x.name.startsWith(`blocks.${l}.`) || x.name.startsWith(`b${l}.`) || x.name.startsWith(`h.${l}.`) || x.name.startsWith(`layers.${l}.`)) });
    const used = new Set(out.flatMap((g) => g.items.map((i) => i.name)));
    out.push({ title: "Final", items: t.filter((x) => !used.has(x.name)) });
    return out.filter((g) => g.items.length);
  }

  private renderList(): void {
    const info = this.info!;
    const total = info.tensors.reduce((s, t) => s + t.numel, 0);
    let html = `<div class="w-total"><b>${fmtInt(total)}</b> parameters in <b>${info.tensors.length}</b> tensors<br><span>${info.sourceLabel}</span></div>`;
    for (const g of this.groups()) {
      html += `<div class="w-group">${g.title}</div>`;
      for (const t of g.items) {
        const short = t.name.replace(/^blocks\.\d+\./, "");
        html += `<button class="w-item${t.name === this.selected ? " sel" : ""}" data-name="${t.name}">
          <span class="w-name">${short}</span><span class="w-shape">${t.shape.join("×")}</span>
          ${sparkline(t.hist)}
          <span class="w-std">σ ${fmt(t.std, 3)}</span></button>`;
      }
    }
    this.listEl.innerHTML = html;
    this.listEl.querySelectorAll<HTMLButtonElement>(".w-item").forEach((b) => b.addEventListener("click", () => void this.select(b.dataset.name!)));
  }

  async select(name: string, lens: Lens = "heatmap"): Promise<void> {
    if (!this.info) return;
    this.selected = name;
    this.lens = lens;
    this.renderList();
    this.loading = true;
    let ld = this.cache.get(name);
    if (!ld) {
      const { data, shape } = await getTensor(name);
      ld = buildImage(name, shape, data);
      this.cache.set(name, ld);
    }
    if (this.selected !== name) return;
    this.cur = ld;
    this.loading = false;
    this.pcaPts = null;
    this.cosine = null;
    if (lens === "pca" && name === "tok_emb") this.pcaPts = vocabPca(ld.data, ld.shape[0], ld.shape[1]);
    if (lens === "cosine") this.cosine = cosineImage(ld.data, ld.shape[0], ld.shape[1]);
    this.resetView();
    this.appear.jump(0);
    this.appear.set(1, performance.now(), 800);
    this.renderHeader();
  }

  private resetView(): void {
    const r = this.mainCanvas.getBoundingClientRect();
    const cur = this.cur;
    if (!cur) return;
    const iw = this.lens === "cosine" && this.cosine ? this.cosine.width : cur.cols;
    const ih = this.lens === "cosine" && this.cosine ? this.cosine.height : cur.rows;
    const padL = 56, padT = 26, padR = 80, padB = 34;
    const zoom = Math.min((r.width - padL - padR) / iw, (r.height - padT - padB) / ih);
    this.fit = { zoom, x: padL + (r.width - padL - padR - iw * zoom) / 2, y: padT + (r.height - padT - padB - ih * zoom) / 2 };
    this.view = { ...this.fit };
  }

  private renderHeader(): void {
    const cur = this.cur!;
    const s = this.info!.tensors.find((t) => t.name === cur.name)!;
    const frob = Math.sqrt(s.numel * (s.std * s.std + s.mean * s.mean));
    this.headerEl.innerHTML = `
      <div class="w-title"><span class="w-tname">${cur.name}</span><span class="w-tshape">${cur.shape.join(" × ")}</span>
        ${cur.transposed ? `<span class="w-note">shown transposed: rows = the ${cur.shape[1]} features</span>` : ""}</div>
      <div class="w-stats">
        <span>params <b>${fmtInt(s.numel)}</b></span><span>mean <b>${fmt(s.mean, 4)}</b></span><span>std <b>${fmt(s.std, 4)}</b></span>
        <span>min <b>${fmt(s.min, 3)}</b></span><span>max <b>${fmt(s.max, 3)}</b></span><span>‖W‖<sub>F</sub> <b>${fmt(frob, 2)}</b></span>
        <span>|w|&lt;1e-3 <b>${pct(s.nearZero)}</b></span></div>`;
    const lenses: { id: Lens; label: string }[] = [{ id: "heatmap", label: "Heatmap" }];
    if (cur.name === "pos_emb" || cur.name === "tok_emb") lenses.push({ id: "cosine", label: cur.name === "pos_emb" ? "Cosine similarity P·Pᵀ" : "Cosine similarity (first 256 ids)" });
    if (cur.name === "tok_emb") lenses.push({ id: "pca", label: "Vocabulary PCA" });
    this.lensEl.innerHTML = lenses.map((l) => `<button class="chip${l.id === this.lens ? " on" : ""}" data-lens="${l.id}">${l.label}</button>`).join("") +
      `<span class="w-hint">wheel: zoom · drag: pan · double-click: reset</span>`;
    this.lensEl.querySelectorAll<HTMLButtonElement>("[data-lens]").forEach((b) => b.addEventListener("click", () => void this.select(cur.name, b.dataset.lens as Lens)));
  }

  frame(now: number): void {
    this.drawMain(now);
    this.drawSide(now);
  }

  private drawMain(now: number): void {
    const { w, h, dpr } = fitCanvas(this.mainCanvas);
    const ctx = this.mainCanvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    resetCtx(ctx);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);
    const cur = this.cur;
    if (!cur) {
      ctx.fillStyle = C.dim;
      ctx.font = `14px ${FONT_SERIF}`;
      ctx.fillText(this.loading ? "loading tensor…" : "select a tensor", 24, 40);
      return;
    }
    if (this.fit.zoom === 1 && this.view.zoom === 1) this.resetView();
    const a = easeOutCubic(this.appear.get(now));
    if (this.lens === "pca" && this.pcaPts) { this.drawPca(ctx, w, h, a); return; }
    const img = this.lens === "cosine" && this.cosine ? this.cosine : cur.image;
    const { zoom, x, y } = this.view;
    const iw = img.width, ih = img.height;
    const plotR = w - 84, plotB = h - 26;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, plotR, plotB);
    ctx.clip();
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    // reveal: the matrix wipes in from the top
    ctx.beginPath();
    ctx.rect(x, y, iw * zoom, ih * zoom * a);
    ctx.clip();
    ctx.drawImage(img, x, y, iw * zoom, ih * zoom);
    ctx.restore();
    ctx.strokeStyle = C.line;
    ctx.strokeRect(x - 0.5, y - 0.5, iw * zoom + 1, ih * zoom + 1);
    // per-head separators for the fused QKV projection
    if (this.lens === "heatmap" && /attn\.qkv\.W$/.test(cur.name) && this.info) {
      const H = this.info.config.nHeads, d = this.info.config.dModel, dh = d / H;
      for (let r = 0; r <= 3 * H; r++) {
        const yy = y + r * dh * zoom;
        ctx.strokeStyle = r % H === 0 ? withAlpha(C.yellow, 0.7) : withAlpha(C.text, 0.18);
        ctx.lineWidth = r % H === 0 ? 1.4 : 0.8;
        ctx.beginPath(); ctx.moveTo(x - 6, yy); ctx.lineTo(x + iw * zoom, yy); ctx.stroke();
        if (r < 3 * H) {
          ctx.fillStyle = HEAD_COLORS[r % H];
          ctx.font = `10px ${FONT_SANS}`;
          ctx.textAlign = "right";
          ctx.textBaseline = "middle";
          ctx.fillText(`${"QKV"[Math.floor(r / H)]}${(r % H) + 1}`, x - 8, yy + (dh * zoom) / 2);
          ctx.textAlign = "left";
        }
      }
    }
    // grid lines and numbers once a cell is large enough (matrix-with-numbers look)
    const cellPx = zoom;
    const src = this.lens === "cosine" && this.cosine ? null : cur;
    if (cellPx >= 9) {
      const c0 = Math.max(0, Math.floor(-x / zoom)), c1 = Math.min(iw, Math.ceil((w - x) / zoom));
      const r0 = Math.max(0, Math.floor(-y / zoom)), r1 = Math.min(ih, Math.ceil((h - y) / zoom));
      ctx.strokeStyle = "rgba(14,17,22,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let c = c0; c <= c1; c++) { ctx.moveTo(x + c * zoom, y + r0 * zoom); ctx.lineTo(x + c * zoom, y + r1 * zoom); }
      for (let r = r0; r <= r1; r++) { ctx.moveTo(x + c0 * zoom, y + r * zoom); ctx.lineTo(x + c1 * zoom, y + r * zoom); }
      ctx.stroke();
      if (cellPx >= 34 && src && (c1 - c0) * (r1 - r0) < 2500) {
        ctx.font = `${Math.min(13, cellPx * 0.28)}px ${FONT_MONO}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) {
          const v = valueAt(src, r, c);
          ctx.fillStyle = Math.abs(v) / src.scale > 0.55 ? "#0e1116" : C.text;
          ctx.fillText(v.toFixed(2), x + (c + 0.5) * zoom, y + (r + 0.5) * zoom);
        }
        ctx.textAlign = "left";
      }
    }
    ctx.restore();
    ctx.fillStyle = C.bg;
    ctx.fillRect(plotR, 0, w - plotR, h);
    ctx.fillRect(0, plotB, w, h - plotB);
    // axes labels
    ctx.font = `italic 12px ${FONT_SERIF}`;
    ctx.fillStyle = C.dim;
    ctx.textBaseline = "alphabetic";
    const rl = this.lens === "cosine" ? "position i" : axisName(cur, "rows");
    const cl = this.lens === "cosine" ? "position j" : axisName(cur, "cols");
    ctx.fillText(`rows: ${rl} (${ih})   ·   columns: ${cl} (${iw})   ·   zoom ${fmt(zoom / this.fit.zoom, 1)}×`, 56, h - 9);
    // colour bar
    this.colorBar(ctx, w - 66, 30, 14, Math.min(260, h - 90), this.lens === "cosine" ? 1 : cur.scale, this.lens === "cosine" ? "cos" : "w");
    // hover readout
    const ms = this.mouse;
    if (ms) {
      const c = Math.floor((ms.x - x) / zoom), r = Math.floor((ms.y - y) / zoom);
      if (c >= 0 && r >= 0 && c < iw && r < ih) {
        ctx.strokeStyle = withAlpha(C.yellow, 0.9);
        ctx.lineWidth = 1;
        ctx.strokeRect(x + c * zoom - 0.5, y + r * zoom - 0.5, Math.max(2, zoom) + 1, Math.max(2, zoom) + 1);
        ctx.strokeStyle = withAlpha(C.yellow, 0.18);
        ctx.beginPath();
        ctx.moveTo(x, y + (r + 0.5) * zoom); ctx.lineTo(x + iw * zoom, y + (r + 0.5) * zoom);
        ctx.moveTo(x + (c + 0.5) * zoom, y); ctx.lineTo(x + (c + 0.5) * zoom, y + ih * zoom);
        ctx.stroke();
        if (this.lens === "cosine" && this.cosine) {
          const n = cur.shape[0];
          const N = Math.min(n, 256);
          const ci = cosAt(cur.data, cur.shape[1], r, c);
          void N;
          tooltip(ctx, ms.x, ms.y, [`cos(${cur.name}[${r}], ${cur.name}[${c}])`, `= ${fmt(ci, 4)}`], w);
        } else {
          const [i, j] = cur.transposed ? [c, r] : [r, c];
          const v = valueAt(cur, r, c);
          const lines = [`${cur.name}[${i}, ${j}] = ${v.toPrecision(6)}`];
          if (cur.name === "tok_emb") lines.push(`token ${i}: '${showToken(this.vocabTexts[i] ?? "?")}', feature ${j}`);
          else if (cur.name === "pos_emb") lines.push(`position ${i}, feature ${j}`);
          lines.push(`row ‖·‖ = ${fmt(cur.rowNorms[r], 3)}`);
          tooltip(ctx, ms.x, ms.y, lines, w);
        }
      }
    }
  }

  private drawPca(ctx: CanvasRenderingContext2D, w: number, h: number, a: number): void {
    const pts = this.pcaPts!;
    const n = pts.length / 2;
    // robust scale (97th percentile); the few outliers are pinned to the rim
    const absv = Array.from(pts, Math.abs).sort((x, y) => x - y);
    const mx = absv[Math.floor(absv.length * 0.95)] || 1;
    const cx = w / 2, cy = h / 2 + 6, R = Math.min(w, h) / 2 - 40;
    const sc = (R / mx) * a;
    const pin = (x: number, y: number): [number, number] => {
      const r = Math.hypot(x, y);
      return r > R ? [(x / r) * R, (y / r) * R] : [x, y];
    };
    ctx.strokeStyle = C.grid;
    ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
    ctx.font = `italic 12px ${FONT_SERIF}`;
    ctx.fillStyle = C.dim;
    ctx.fillText("PC1", cx + R - 24, cy - 6);
    ctx.fillText("PC2", cx + 6, cy - R + 12);
    let hover = -1, hd = 64;
    ctx.textBaseline = "alphabetic";
    const P = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      const [px, py] = pin(pts[i * 2] * sc, pts[i * 2 + 1] * sc);
      P[i * 2] = cx + px; P[i * 2 + 1] = cy - py;
    }
    for (let i = 0; i < n; i++) {
      const x = P[i * 2], y = P[i * 2 + 1];
      const t = this.vocabTexts[i] ?? "";
      const kw = JAVA_KEYWORDS.has(t.trim());
      ctx.fillStyle = kw ? C.yellow : /^\s+$/.test(t) ? C.teal : /^[A-Za-z_]/.test(t.trim()) ? withAlpha(C.blue, 0.55) : withAlpha(C.red, 0.55);
      ctx.fillRect(x - (kw ? 2 : 1), y - (kw ? 2 : 1), kw ? 4 : 2, kw ? 4 : 2);
      if (this.mouse) {
        const d = (this.mouse.x - x) ** 2 + (this.mouse.y - y) ** 2;
        if (d < hd) { hd = d; hover = i; }
      }
    }
    // keyword labels, skipping any that would overlap one already placed
    ctx.font = `11px ${FONT_MONO}`;
    const placed: [number, number, number, number][] = [];
    for (let i = 0; i < n; i++) {
      const t = (this.vocabTexts[i] ?? "").trim();
      if (!JAVA_KEYWORDS.has(t)) continue;
      const x = P[i * 2] + 4, y = P[i * 2 + 1] - 4, tw = ctx.measureText(t).width;
      if (placed.some(([a0, b0, a1, b1]) => x < a1 && x + tw > a0 && y - 10 < b1 && y > b0)) continue;
      placed.push([x, y - 10, x + tw, y]);
      ctx.fillStyle = C.yellow;
      ctx.fillText(t, x, y);
    }
    // legend
    const lg: [string, string][] = [[C.yellow, "Java keywords / common types"], [withAlpha(C.blue, 0.8), "identifier pieces"], [C.teal, "whitespace"], [withAlpha(C.red, 0.8), "symbols"]];
    ctx.font = `11px ${FONT_SANS}`;
    ctx.textBaseline = "alphabetic";
    lg.forEach(([c, l], i) => { ctx.fillStyle = c; ctx.fillRect(18, 18 + i * 17, 8, 8); ctx.fillStyle = C.dim; ctx.fillText(l, 32, 26 + i * 17); });
    ctx.fillStyle = C.faint;
    ctx.fillText(`${n} token embeddings projected on their top-2 principal components (power iteration in the browser); points beyond the 95th percentile of |coordinate| sit on the rim`, 18, h - 12);
    if (hover >= 0 && this.mouse) tooltip(ctx, this.mouse.x, this.mouse.y, [`token ${hover}: '${showToken(this.vocabTexts[hover] ?? "?")}'`, `PC = (${fmt(pts[hover * 2], 3)}, ${fmt(pts[hover * 2 + 1], 3)})`], w);
  }

  private colorBar(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, scale: number, sym: string): void {
    for (let i = 0; i < h; i++) {
      const v = 1 - (2 * i) / h;
      ctx.fillStyle = divColor(v, 1);
      ctx.fillRect(x, y + i, w, 1.2);
    }
    ctx.strokeStyle = C.line;
    ctx.strokeRect(x - 0.5, y - 0.5, w + 1, h + 1);
    ctx.font = `10.5px ${FONT_MONO}`;
    ctx.fillStyle = C.dim;
    ctx.textBaseline = "middle";
    ctx.fillText(`+${fmt(scale, 2)}`, x + w + 4, y + 4);
    ctx.fillText("0", x + w + 4, y + h / 2);
    ctx.fillText(`−${fmt(scale, 2)}`, x + w + 4, y + h - 4);
    ctx.font = `italic 12px ${FONT_SERIF}`;
    ctx.fillText(sym, x + 2, y - 12);
  }

  private drawSide(now: number): void {
    const { w, h, dpr } = fitCanvas(this.sideCanvas);
    const ctx = this.sideCanvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    resetCtx(ctx);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);
    if (!this.info || !this.cur) return;
    const s = this.info.tensors.find((t) => t.name === this.cur!.name)!;
    const a = easeOutCubic(this.appear.get(now));
    const pad = 14;
    // histogram
    const hh = Math.min(220, h * 0.32);
    ctx.font = `600 12px ${FONT_SANS}`;
    ctx.fillStyle = C.text;
    ctx.textBaseline = "alphabetic";
    ctx.fillText("Histogram (32 bins)", pad, 20);
    const hx = pad, hy = 30, hw = w - pad * 2, hhh = hh - 40;
    const mx = Math.max(...s.hist, 1);
    const bw = hw / s.hist.length;
    s.hist.forEach((c, i) => {
      const v0 = s.min + ((s.max - s.min) * (i + 0.5)) / s.hist.length;
      const bh = (c / mx) * hhh * clamp01(a * 1.4 - i / s.hist.length * 0.4);
      ctx.fillStyle = divColor(v0, Math.max(Math.abs(s.min), Math.abs(s.max)) * 0.9);
      ctx.fillRect(hx + i * bw + 0.5, hy + hhh - bh, bw - 1, bh);
    });
    ctx.strokeStyle = C.line;
    ctx.beginPath(); ctx.moveTo(hx, hy + hhh + 0.5); ctx.lineTo(hx + hw, hy + hhh + 0.5); ctx.stroke();
    if (s.min < 0 && s.max > 0) {
      const zx = hx + ((0 - s.min) / (s.max - s.min)) * hw;
      ctx.strokeStyle = withAlpha(C.yellow, 0.6);
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(zx, hy); ctx.lineTo(zx, hy + hhh); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.font = `10.5px ${FONT_MONO}`;
    ctx.fillStyle = C.dim;
    ctx.fillText(fmt(s.min, 3), hx, hy + hhh + 14);
    ctx.textAlign = "right";
    ctx.fillText(fmt(s.max, 3), hx + hw, hy + hhh + 14);
    ctx.textAlign = "left";

    // per-layer norms of each tensor kind
    const ny = hh + 36;
    const nh = Math.min(260, h * 0.36);
    ctx.font = `600 12px ${FONT_SANS}`;
    ctx.fillStyle = C.text;
    ctx.fillText("Per-layer norms ‖W‖_F", pad, ny);
    const L = this.info.config.nLayers;
    const kinds = ["attn.qkv.W", "attn.proj.W", "mlp.fc.W", "mlp.proj.W", "ln1.g", "ln2.g"];
    const kindCol = [C.blue, C.teal, C.yellow, C.orange, C.purple, C.pink];
    const series = kinds.map((k) => Array.from({ length: L }, (_, l) => {
      const t = this.info!.tensors.find((x) => x.name === `blocks.${l}.${k}` || (x.name.endsWith(`.${k}`) && x.name.includes(`${l}.`)));
      return t ? Math.sqrt(t.numel * (t.std * t.std + t.mean * t.mean)) : 0;
    }));
    const nmax = Math.max(...series.flat(), 1e-6);
    const px0 = pad + 30, px1 = w - pad - 8, py0 = ny + 14, py1 = ny + nh - 50;
    ctx.strokeStyle = C.grid;
    ctx.font = `10px ${FONT_MONO}`;
    ctx.fillStyle = C.faint;
    for (let i = 0; i <= 4; i++) {
      const yy = py1 - ((py1 - py0) * i) / 4;
      ctx.beginPath(); ctx.moveTo(px0, yy); ctx.lineTo(px1, yy); ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(fmt((nmax * i) / 4, 0), px0 - 4, yy + 3);
    }
    ctx.textAlign = "center";
    for (let l = 0; l < L; l++) ctx.fillText(`L${l + 1}`, px0 + ((px1 - px0) * l) / Math.max(1, L - 1), py1 + 14);
    ctx.textAlign = "left";
    series.forEach((ser, k) => {
      const isSel = this.cur!.name.endsWith(kinds[k]);
      ctx.strokeStyle = withAlpha(kindCol[k], isSel ? 1 : 0.6);
      ctx.lineWidth = isSel ? 2.4 : 1.3;
      ctx.beginPath();
      ser.forEach((v, l) => {
        const xx = px0 + ((px1 - px0) * l) / Math.max(1, L - 1), yy = py1 - ((py1 - py0) * v * a) / nmax;
        if (l === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
      });
      ctx.stroke();
      ser.forEach((v, l) => {
        const xx = px0 + ((px1 - px0) * l) / Math.max(1, L - 1), yy = py1 - ((py1 - py0) * v * a) / nmax;
        ctx.fillStyle = kindCol[k];
        ctx.beginPath(); ctx.arc(xx, yy, isSel ? 3.2 : 2.2, 0, Math.PI * 2); ctx.fill();
      });
    });
    ctx.font = `10.5px ${FONT_MONO}`;
    kinds.forEach((k, i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const lx = pad + col * ((w - pad * 2) / 2), ly = py1 + 30 + row * 15;
      ctx.fillStyle = kindCol[i];
      ctx.fillRect(lx, ly - 7, 8, 8);
      ctx.fillStyle = this.cur!.name.endsWith(k) ? C.text : C.dim;
      ctx.fillText(k, lx + 12, ly);
    });

    // row-norm profile of the selected tensor
    const ry = ny + nh + 34;
    if (ry + 60 < h) {
      ctx.font = `600 12px ${FONT_SANS}`;
      ctx.fillStyle = C.text;
      ctx.fillText(`Row norms (${this.cur.rows} rows as shown)`, pad, ry);
      const rn = this.cur.rowNorms;
      let rmx = 0;
      for (let i = 0; i < rn.length; i++) rmx = Math.max(rmx, rn[i]);
      const rx0 = pad, rx1 = w - pad, ry0 = ry + 10, ry1 = Math.min(h - 14, ry + 120);
      const n = rn.length;
      ctx.fillStyle = withAlpha(C.teal, 0.8);
      const step = Math.max(1, Math.ceil(n / (rx1 - rx0)));
      for (let i = 0; i < n; i += step) {
        let v = 0;
        for (let j = i; j < Math.min(n, i + step); j++) v = Math.max(v, rn[j]);
        const xx = rx0 + ((rx1 - rx0) * i) / n;
        const bh = ((ry1 - ry0) * v * a) / (rmx || 1);
        ctx.fillRect(xx, ry1 - bh, Math.max(1, ((rx1 - rx0) * step) / n - 0.3), bh);
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------

function axisName(cur: Loaded, which: "rows" | "cols"): string {
  const n = cur.name;
  const s = cur.shape;
  const fromShape = (dim: number) => {
    if (n === "tok_emb") return dim === 0 ? "token id" : "feature";
    if (n === "pos_emb") return dim === 0 ? "position" : "feature";
    if (/qkv\.W$/.test(n)) return dim === 0 ? "output (Q|K|V × heads)" : "input feature";
    if (/fc\.W$/.test(n)) return dim === 0 ? "MLP neuron" : "input feature";
    if (/proj\.W$/.test(n)) return dim === 0 ? "output feature" : n.includes("mlp") ? "MLP neuron" : "input feature";
    return s.length === 1 ? (dim === 0 ? "" : "index") : `dim ${dim}`;
  };
  if (cur.shape.length === 1) return which === "rows" ? "-" : "index";
  const dim = which === "rows" ? (cur.transposed ? 1 : 0) : cur.transposed ? 0 : 1;
  return fromShape(dim);
}

function valueAt(cur: Loaded, r: number, c: number): number {
  const C0 = cur.shape.length >= 2 ? cur.data.length / cur.shape[0] : cur.data.length;
  if (cur.shape.length === 1) return cur.data[c];
  return cur.transposed ? cur.data[c * C0 + r] : cur.data[r * C0 + c];
}

function robustScale(data: Float32Array): number {
  const n = data.length;
  const step = Math.max(1, Math.floor(n / 20000));
  const a: number[] = [];
  for (let i = 0; i < n; i += step) a.push(Math.abs(data[i]));
  a.sort((x, y) => x - y);
  return a[Math.floor(a.length * 0.99)] || a[a.length - 1] || 1;
}

function buildImage(name: string, shape: number[], data: Float32Array): Loaded {
  let R = shape.length >= 2 ? shape[0] : 1;
  let Cc = data.length / R;
  const transposed = shape.length >= 2 && R / Cc > 4;
  const rows = transposed ? Cc : R, cols = transposed ? R : Cc;
  const scale = robustScale(data);
  const img = document.createElement("canvas");
  img.width = cols;
  img.height = rows;
  const ictx = img.getContext("2d")!;
  const id = ictx.createImageData(cols, rows);
  const rowNorms = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let c = 0; c < cols; c++) {
      const v = transposed ? data[c * Cc + r] : data[r * Cc + c];
      s += v * v;
      const k = Math.round((Math.max(-1, Math.min(1, v / scale)) + 1) * 127.5);
      const o = (r * cols + c) * 4;
      id.data[o] = DIV_LUT[k * 3];
      id.data[o + 1] = DIV_LUT[k * 3 + 1];
      id.data[o + 2] = DIV_LUT[k * 3 + 2];
      id.data[o + 3] = 255;
    }
    rowNorms[r] = Math.sqrt(s);
  }
  ictx.putImageData(id, 0, 0);
  void R; void Cc;
  R = rows; Cc = cols;
  return { name, shape, data, rows, cols, transposed, scale, image: img, rowNorms };
}

function cosAt(data: Float32Array, d: number, i: number, j: number): number {
  let s = 0, a = 0, b = 0;
  for (let k = 0; k < d; k++) {
    const x = data[i * d + k], y = data[j * d + k];
    s += x * y; a += x * x; b += y * y;
  }
  return s / (Math.sqrt(a * b) || 1);
}

function cosineImage(data: Float32Array, n: number, d: number): HTMLCanvasElement {
  const N = Math.min(n, 256);
  const img = document.createElement("canvas");
  img.width = N;
  img.height = N;
  const ctx = img.getContext("2d")!;
  const id = ctx.createImageData(N, N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const v = cosAt(data, d, i, j);
    const k = Math.round((Math.max(-1, Math.min(1, v)) + 1) * 127.5);
    const o = (i * N + j) * 4;
    id.data[o] = DIV_LUT[k * 3]; id.data[o + 1] = DIV_LUT[k * 3 + 1]; id.data[o + 2] = DIV_LUT[k * 3 + 2]; id.data[o + 3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  return img;
}

/** Top-2 principal components of the rows of [n, d] by power iteration with deflation. */
function vocabPca(data: Float32Array, n: number, d: number): Float32Array {
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += data[i * d + j] / n;
  const comps: Float64Array[] = [];
  for (let c = 0; c < 2; c++) {
    let v = new Float64Array(d).map((_, j) => Math.sin(j * 1.7 + c * 3.1) + 0.01);
    for (let it = 0; it < 60; it++) {
      const nv = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        let dot = 0;
        for (let j = 0; j < d; j++) dot += (data[i * d + j] - mean[j]) * v[j];
        for (let j = 0; j < d; j++) nv[j] += dot * (data[i * d + j] - mean[j]);
      }
      for (const u of comps) {
        let p = 0;
        for (let j = 0; j < d; j++) p += nv[j] * u[j];
        for (let j = 0; j < d; j++) nv[j] -= p * u[j];
      }
      let norm = 0;
      for (let j = 0; j < d; j++) norm += nv[j] * nv[j];
      norm = Math.sqrt(norm) || 1;
      v = nv.map((x) => x / norm);
    }
    comps.push(v);
  }
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) for (let c = 0; c < 2; c++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += (data[i * d + j] - mean[j]) * comps[c][j];
    out[i * 2 + c] = s;
  }
  return out;
}

function sparkline(hist: number[]): string {
  const mx = Math.max(...hist, 1);
  const w = 64, h = 16;
  const bw = w / hist.length;
  const bars = hist.map((c, i) => {
    const bh = Math.max(0.5, (c / mx) * h);
    return `<rect x="${(i * bw).toFixed(1)}" y="${(h - bh).toFixed(1)}" width="${(bw - 0.4).toFixed(1)}" height="${bh.toFixed(1)}"/>`;
  }).join("");
  return `<svg class="w-spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">${bars}</svg>`;
}

void SEQ_LUT;
void roundRect;
