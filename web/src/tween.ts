/**
 * A tiny tween engine. Nothing in the lab snaps: when new activations arrive, every displayed
 * number interpolates from where it currently is to the new target (ease-in-out cubic).
 */
export const easeInOutCubic = (t: number): number => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const easeOutCubic = (t: number): number => 1 - Math.pow(1 - t, 3);
export const clamp01 = (t: number): number => (t < 0 ? 0 : t > 1 ? 1 : t);
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

export class TweenNum {
  private from: number;
  private to: number;
  private t0 = -1e9;
  constructor(v = 0, public dur = 550, public ease: (t: number) => number = easeInOutCubic) {
    this.from = v;
    this.to = v;
  }
  set(v: number, now: number, dur = this.dur): void {
    if (v === this.to) return;
    this.from = this.get(now);
    this.to = v;
    this.t0 = now;
    this.dur = dur;
  }
  jump(v: number): void {
    this.from = this.to = v;
    this.t0 = -1e9;
  }
  get(now: number): number {
    const t = clamp01((now - this.t0) / this.dur);
    return t >= 1 ? this.to : lerp(this.from, this.to, this.ease(t));
  }
  get target(): number {
    return this.to;
  }
  active(now: number): boolean {
    return now - this.t0 < this.dur;
  }
}

export class TweenVec {
  private from: Float32Array;
  private to: Float32Array;
  private cur: Float32Array;
  private t0 = -1e9;
  constructor(n: number, public dur = 550) {
    this.from = new Float32Array(n);
    this.to = new Float32Array(n);
    this.cur = new Float32Array(n);
  }
  get length(): number {
    return this.to.length;
  }
  set(v: ArrayLike<number>, now: number, dur = this.dur): void {
    const cur = this.get(now);
    if (v.length !== this.to.length) {
      this.from = new Float32Array(v.length);
      this.to = new Float32Array(v.length);
      this.cur = new Float32Array(v.length);
      this.from.set(cur.subarray(0, Math.min(cur.length, v.length)));
    } else {
      this.from.set(cur);
    }
    for (let i = 0; i < v.length; i++) this.to[i] = v[i];
    this.t0 = now;
    this.dur = dur;
  }
  jump(v: ArrayLike<number>): void {
    if (v.length !== this.to.length) {
      this.from = new Float32Array(v.length);
      this.to = new Float32Array(v.length);
      this.cur = new Float32Array(v.length);
    }
    for (let i = 0; i < v.length; i++) this.from[i] = this.to[i] = this.cur[i] = v[i];
    this.t0 = -1e9;
  }
  get(now: number): Float32Array {
    const t = clamp01((now - this.t0) / this.dur);
    if (t >= 1) return this.to;
    const e = easeInOutCubic(t);
    const { from, to, cur } = this;
    for (let i = 0; i < to.length; i++) cur[i] = from[i] + (to[i] - from[i]) * e;
    return cur;
  }
  get target(): Float32Array {
    return this.to;
  }
}

/** Partial cubic Bezier (de Casteljau split at t) so curves can "draw themselves on". */
export function bezierPartial(
  ctx: CanvasRenderingContext2D,
  x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number, t: number,
): void {
  if (t <= 0) return;
  if (t >= 1) {
    ctx.moveTo(x0, y0);
    ctx.bezierCurveTo(x1, y1, x2, y2, x3, y3);
    return;
  }
  const ax = lerp(x0, x1, t), ay = lerp(y0, y1, t);
  const bx = lerp(x1, x2, t), by = lerp(y1, y2, t);
  const cx = lerp(x2, x3, t), cy = lerp(y2, y3, t);
  const dx = lerp(ax, bx, t), dy = lerp(ay, by, t);
  const ex = lerp(bx, cx, t), ey = lerp(by, cy, t);
  const fx = lerp(dx, ex, t), fy = lerp(dy, ey, t);
  ctx.moveTo(x0, y0);
  ctx.bezierCurveTo(ax, ay, dx, dy, fx, fy);
}

/** Resize a canvas to its CSS box at device-pixel resolution; returns CSS width/height. */
export function fitCanvas(canvas: HTMLCanvasElement): { w: number; h: number; dpr: number } {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  return { w, h, dpr };
}
