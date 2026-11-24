/** Types and clients for the local inference server (docs/SERVER.md). */

export interface ModelConfig {
  vocabSize: number;
  ctx: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  dMlp: number;
}

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
  nearZero: number;
  hist: number[];
}

export interface ModelInfo {
  source: { kind: string; dir?: string; run?: string; reason?: string };
  sourceLabel: string;
  warnings: string[];
  config: ModelConfig;
  numParams: number;
  kernel: string;
  saved?: Record<string, unknown>;
  tokenizer: { type: string; vocabSize: number; numMerges: number; pretokenRegex: string };
  pca: { components: number; explained: number[]; fitRows: number };
  tensors: TensorStats[];
}

export interface TokenView {
  id: number;
  text: string;
}

export interface TopK {
  id: number;
  text: string;
  p: number;
  logit: number;
}

export interface StepInfo {
  i: number;
  id: number;
  text: string;
  p: number;
  entropy: number;
  topk: TopK[];
}

export interface TracePayload {
  reqId: number;
  ms: number;
  T: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  dMlp: number;
  stages: string[];
  pca: number[][];
  norms: number[][];
  lastVectors?: number[][];
  attention: { last: number[][][]; windowStart: number; window: number; matrices: number[][][] };
  mlp: {
    lastTop: { i: number; v: number }[][];
    argmax: number[][];
    argmaxV: number[][];
    activeFrac: number[];
    lastAll?: number[][];
    windowTop?: { i: number; v: number }[][][];
  };
  logitLens: { topk: TopK[]; entropy: number }[];
  next: { topk: TopK[]; entropy: number };
}

export interface TokensMsg {
  reqId: number;
  tokens: TokenView[];
  truncated: number;
  prefixChars: number;
}

export interface CompleteResult {
  completion: string;
  stopReason: string;
  prompt: { tokens: TokenView[]; truncated: number; prefixChars: number };
  steps: StepInfo[];
  ms: number;
  model: string;
}

export async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return (await r.json()) as T;
}

/** Raw Float32 tensor (exact values, used for the weights browser and embedding columns). */
export async function getTensor(name: string): Promise<{ data: Float32Array; shape: number[] }> {
  const r = await fetch(`/api/weights/${encodeURIComponent(name)}?format=bin`);
  if (!r.ok) throw new Error(`tensor ${name}: HTTP ${r.status}`);
  const shape = (r.headers.get("x-shape") ?? "").split(",").filter(Boolean).map(Number);
  return { data: new Float32Array(await r.arrayBuffer()), shape };
}

/**
 * Ghost-text completions over POST /api/complete. A newer request aborts the older one; the
 * server notices the dropped connection and stops generating (see http.ts).
 */
export class Completer {
  private ctrl: AbortController | null = null;
  private seq = 0;
  async complete(code: string, cursor: number, maxTokens = 24): Promise<CompleteResult | null> {
    this.ctrl?.abort();
    const ctrl = new AbortController();
    this.ctrl = ctrl;
    const my = ++this.seq;
    try {
      const r = await fetch("/api/complete", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ code, cursor, maxTokens, topK: 10, temperature: 0, stopAtNewline: true }),
        signal: ctrl.signal,
      });
      if (!r.ok) return null;
      const j = (await r.json()) as CompleteResult;
      return my === this.seq ? j : null;
    } catch {
      return null;
    }
  }
  cancel(): void {
    this.seq++;
    this.ctrl?.abort();
    this.ctrl = null;
  }
}

export interface TraceHandlers {
  onHello?(m: { model: string; config: ModelConfig; warnings: string[] }): void;
  /** A trace with the tokens it was computed on. */
  onTrace?(m: TracePayload, tokens: TokensMsg): void;
  onDone?(m: { reqId: number; traceMs: number; totalMs: number }): void;
  onCancelled?(reqId: number): void;
  onStatus?(s: "connecting" | "open" | "closed"): void;
}

/**
 * WS /api/trace client. The server debounces (60 ms) and cancels requests that a newer keystroke
 * made stale. The client renders a trace only if it is newer than the last one it rendered, so
 * out-of-order or superseded frames are dropped. While you type continuously, the newest finished
 * pass is still shown: it is never more than one forward pass behind the keyboard.
 */
export class TraceClient {
  private ws: WebSocket | null = null;
  private reqId = 0;
  private rendered = 0;
  private pending: string | null = null;
  private tokens = new Map<number, TokensMsg>();
  stats = { sent: 0, traces: 0, cancelled: 0, dropped: 0, lastMs: 0, behind: 0 };
  constructor(private h: TraceHandlers) {
    this.connect();
  }
  private connect(): void {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/trace`);
    this.ws = ws;
    this.h.onStatus?.("connecting");
    ws.onopen = () => {
      this.h.onStatus?.("open");
      if (this.pending) ws.send(this.pending);
      this.pending = null;
    };
    ws.onclose = () => {
      this.h.onStatus?.("closed");
      setTimeout(() => this.connect(), 1200);
    };
    ws.onmessage = (ev) => {
      const m = JSON.parse(String(ev.data)) as { type: string; reqId?: number } & Record<string, unknown>;
      if (m.type === "hello") { this.h.onHello?.(m as never); return; }
      const id = Number(m.reqId);
      if (m.reqId !== undefined && id <= this.rendered) { if (m.type === "trace") this.stats.dropped++; return; }
      if (m.type === "tokens") this.tokens.set(id, m as unknown as TokensMsg);
      else if (m.type === "trace") {
        const tk = this.tokens.get(id);
        if (!tk) { this.stats.dropped++; return; }
        this.rendered = id;
        for (const k of this.tokens.keys()) if (k <= id) this.tokens.delete(k);
        this.stats.traces++;
        this.stats.lastMs = Number(m.ms) || 0;
        this.stats.behind = this.reqId - id;
        this.h.onTrace?.(m as unknown as TracePayload, tk);
      } else if (m.type === "done") this.h.onDone?.(m as never);
      else if (m.type === "cancelled") { this.stats.cancelled++; this.h.onCancelled?.(Number(m.reqId)); }
    };
  }
  trace(code: string, cursor: number): number {
    const msg = JSON.stringify({
      type: "trace", reqId: ++this.reqId, code, cursor,
      maxTokens: 0, topK: 10, attnWindow: 16, mlpTopK: 16, mlpWindowTopK: 8, lastVectors: false,
    });
    this.stats.sent++;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(msg);
    else this.pending = msg;
    return this.reqId;
  }
}
