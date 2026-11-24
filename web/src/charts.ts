/** Small canvas line charts in the lab style (axes, nice ticks, draw-on animation, hover crosshair). */
import { C, FONT_MONO, FONT_SANS, FONT_SERIF, fmt, withAlpha } from "./theme";
import { fitCanvas } from "./tween";

export interface Series {
  label: string;
  color: string;
  points: [number, number][];
  dots?: boolean;
  width?: number;
  dash?: number[];
}

export interface ChartOpts {
  title: string;
  xLabel: string;
  yLabel?: string;
  logY?: boolean;
  xFmt?: (v: number) => string;
  yFmt?: (v: number) => string;
  note?: string;
}

function niceTicks(lo: number, hi: number, n = 5): number[] {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

export class LineChart {
  private mouse: { x: number; y: number } | null = null;
  constructor(readonly canvas: HTMLCanvasElement, public series: Series[], public opts: ChartOpts) {
    canvas.addEventListener("mousemove", (e) => {
      const r = canvas.getBoundingClientRect();
      this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
    });
    canvas.addEventListener("mouseleave", () => { this.mouse = null; });
  }

  draw(progress: number): void {
    const { w, h, dpr } = fitCanvas(this.canvas);
    const ctx = this.canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.setLineDash([]);
    const o = this.opts;
    const padL = 52, padR = 14, padT = 30, padB = 34;
    ctx.font = `600 12px ${FONT_SANS}`;
    ctx.fillStyle = C.text;
    ctx.textBaseline = "alphabetic";
    ctx.fillText(o.title, 10, 18);
    // legend
    ctx.font = `11px ${FONT_SANS}`;
    let lx = w - padR;
    for (const s of [...this.series].reverse()) {
      if (!s.points.length) continue;
      const tw = ctx.measureText(s.label).width;
      lx -= tw;
      ctx.fillStyle = C.dim;
      ctx.fillText(s.label, lx, 18);
      lx -= 14;
      ctx.fillStyle = s.color;
      ctx.fillRect(lx, 12, 9, 3);
      lx -= 12;
    }
    const all = this.series.flatMap((s) => s.points);
    if (!all.length) {
      ctx.font = `italic 13px ${FONT_SERIF}`;
      ctx.fillStyle = C.faint;
      ctx.fillText(o.note ?? "no data yet", padL, h / 2);
      return;
    }
    const tf = (v: number) => (o.logY ? Math.log10(Math.max(v, 1e-12)) : v);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of all) {
      if (!Number.isFinite(y)) continue;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x);
      y0 = Math.min(y0, tf(y)); y1 = Math.max(y1, tf(y));
    }
    if (x1 === x0) x1 = x0 + 1;
    const yspan = y1 - y0 || Math.abs(y1) * 0.1 || 1;
    y0 -= yspan * 0.06; y1 += yspan * 0.06;
    const X = (x: number) => padL + ((x - x0) / (x1 - x0)) * (w - padL - padR);
    const Y = (y: number) => h - padB - ((tf(y) - y0) / (y1 - y0)) * (h - padT - padB);
    // grid + ticks
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.font = `10px ${FONT_MONO}`;
    ctx.fillStyle = C.faint;
    const yt = o.logY ? niceTicks(y0, y1, 4).map((v) => Math.pow(10, v)) : niceTicks(y0, y1, 4);
    for (const v of yt) {
      const yy = Y(v);
      if (yy < padT - 2 || yy > h - padB + 2) continue;
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText((o.yFmt ?? ((x: number) => fmt(x, 2)))(v), padL - 6, yy + 3);
    }
    ctx.textAlign = "center";
    for (const v of niceTicks(x0, x1, 5)) {
      const xx = X(v);
      ctx.beginPath(); ctx.moveTo(xx, padT); ctx.lineTo(xx, h - padB); ctx.stroke();
      ctx.fillText((o.xFmt ?? ((x: number) => fmt(x, 0)))(v), xx, h - padB + 13);
    }
    ctx.textAlign = "left";
    ctx.font = `italic 12px ${FONT_SERIF}`;
    ctx.fillStyle = C.dim;
    ctx.fillText(o.xLabel, w / 2 - ctx.measureText(o.xLabel).width / 2, h - 6);
    if (o.yLabel) {
      ctx.save();
      ctx.translate(12, h / 2 + ctx.measureText(o.yLabel).width / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(o.yLabel, 0, 0);
      ctx.restore();
    }
    // series, drawn on from left to right
    const xCut = x0 + (x1 - x0) * progress;
    ctx.save();
    ctx.beginPath();
    ctx.rect(padL, padT - 4, X(xCut) - padL + 1, h - padT - padB + 8);
    ctx.clip();
    for (const s of this.series) {
      if (!s.points.length) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width ?? 1.6;
      ctx.setLineDash(s.dash ?? []);
      ctx.beginPath();
      s.points.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.stroke();
      ctx.setLineDash([]);
      if (s.dots) {
        ctx.fillStyle = s.color;
        for (const [x, y] of s.points) { ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, Math.PI * 2); ctx.fill(); }
      }
    }
    ctx.restore();
    // hover crosshair
    const m = this.mouse;
    if (m && m.x > padL && m.x < w - padR) {
      const xv = x0 + ((m.x - padL) / (w - padL - padR)) * (x1 - x0);
      ctx.strokeStyle = withAlpha(C.text, 0.25);
      ctx.beginPath(); ctx.moveTo(m.x, padT); ctx.lineTo(m.x, h - padB); ctx.stroke();
      const lines: string[] = [`${o.xLabel}: ${(o.xFmt ?? ((x: number) => fmt(x, 0)))(xv)}`];
      for (const s of this.series) {
        if (!s.points.length) continue;
        let best = s.points[0];
        for (const p of s.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
        lines.push(`${s.label}: ${(o.yFmt ?? ((x: number) => fmt(x, 3)))(best[1])}`);
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(X(best[0]), Y(best[1]), 3.5, 0, Math.PI * 2); ctx.fill();
      }
      ctx.font = `11px ${FONT_SANS}`;
      const tw = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 16;
      const tx = m.x + tw + 16 > w ? m.x - tw - 10 : m.x + 10;
      ctx.fillStyle = "rgba(14,17,22,0.94)";
      ctx.fillRect(tx, padT + 4, tw, lines.length * 15 + 8);
      ctx.strokeStyle = C.line;
      ctx.strokeRect(tx, padT + 4, tw, lines.length * 15 + 8);
      ctx.textBaseline = "top";
      lines.forEach((l, i) => { ctx.fillStyle = i ? C.dim : C.text; ctx.fillText(l, tx + 8, padT + 9 + i * 15); });
    }
  }
}
