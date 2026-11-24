/** Visual language: near-black canvas, 3Blue1Brown-like accents, diverging blue/red colormap. */
export const C = {
  bg: "#0e1116",
  panel: "#11151c",
  panel2: "#161b24",
  grid: "#1c232d",
  line: "#29323f",
  text: "#e6edf3",
  dim: "#8b96a5",
  faint: "#566170",
  blue: "#58C4DD",
  blueD: "#29ABCA",
  yellow: "#F7D96F",
  yellowB: "#FFFF66",
  teal: "#5CD0B3",
  red: "#FC6255",
  green: "#83C167",
  purple: "#B189C6",
  orange: "#FF8C42",
  pink: "#E07ACB",
};

export const HEAD_COLORS = [C.blue, C.yellow, C.teal, C.red, C.purple, C.orange, C.green, C.pink];

export const FONT_SANS = '"Inter", "Segoe UI", system-ui, -apple-system, sans-serif';
export const FONT_MONO = '"JetBrains Mono", "Cascadia Code", "Cascadia Mono", Consolas, monospace';
export const FONT_SERIF = '"CMU Serif", "Latin Modern Roman", "Cambria Math", Cambria, Georgia, serif';

type RGB = [number, number, number];

function hex(h: string): RGB {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function mix(a: RGB, b: RGB, t: number): RGB {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

/** 256-entry diverging LUT: index 0 = most negative (red), 128 = zero (near background), 255 = most positive (blue). */
export const DIV_LUT: Uint8ClampedArray = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  const neg2 = hex("#FFD6CF"), neg = hex(C.red), mid = hex("#141a22"), pos = hex(C.blue), pos2 = hex("#DDF6FF");
  for (let i = 0; i < 256; i++) {
    const v = (i - 127.5) / 127.5; // -1..1
    const t = Math.pow(Math.abs(v), 0.8);
    let c: RGB;
    if (v < 0) c = t < 0.85 ? mix(mid, neg, t / 0.85) : mix(neg, neg2, (t - 0.85) / 0.15);
    else c = t < 0.85 ? mix(mid, pos, t / 0.85) : mix(pos, pos2, (t - 0.85) / 0.15);
    lut[i * 3] = c[0]; lut[i * 3 + 1] = c[1]; lut[i * 3 + 2] = c[2];
  }
  return lut;
})();

/** Sequential LUT for probabilities / activations: background -> teal -> blue -> yellow. */
export const SEQ_LUT: Uint8ClampedArray = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  const stops: RGB[] = [hex("#121821"), hex("#1d5560"), hex(C.blue), hex(C.yellow), hex("#FFF8D6")];
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(x));
    const c = mix(stops[k], stops[k + 1], x - k);
    lut[i * 3] = c[0]; lut[i * 3 + 1] = c[1]; lut[i * 3 + 2] = c[2];
  }
  return lut;
})();

const DIV_CSS = Array.from({ length: 256 }, (_, i) => `rgb(${DIV_LUT[i * 3]},${DIV_LUT[i * 3 + 1]},${DIV_LUT[i * 3 + 2]})`);
const SEQ_CSS = Array.from({ length: 256 }, (_, i) => `rgb(${SEQ_LUT[i * 3]},${SEQ_LUT[i * 3 + 1]},${SEQ_LUT[i * 3 + 2]})`);

/** v / scale clamped to [-1, 1] -> css colour. */
export function divColor(v: number, scale: number): string {
  const x = Math.max(-1, Math.min(1, scale > 0 ? v / scale : 0));
  return DIV_CSS[Math.round((x + 1) * 127.5)];
}

export function divIndex(v: number, scale: number): number {
  const x = Math.max(-1, Math.min(1, scale > 0 ? v / scale : 0));
  return Math.round((x + 1) * 127.5);
}

export function seqColor(v: number): string {
  return SEQ_CSS[Math.round(Math.max(0, Math.min(1, v)) * 255)];
}

export function withAlpha(hexColor: string, a: number): string {
  const [r, g, b] = hex(hexColor);
  return `rgba(${r},${g},${b},${Math.max(0, Math.min(1, a)).toFixed(3)})`;
}

/** Make whitespace in a token visible for chips and labels. */
export function showToken(s: string): string {
  if (s === "<|bof|>") return "⟨bof⟩";
  if (s === "<|eof|>") return "⟨eof⟩";
  return s.replace(/\r/g, "").replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/ /g, "·");
}

export function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toFixed(digits);
}

export function fmtInt(v: number): string {
  return Math.round(v).toLocaleString("en-US");
}

export function pct(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}
