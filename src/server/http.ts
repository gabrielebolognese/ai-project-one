/**
 * Local inference server (plain node:http + ws).
 *
 *   GET  /api/health                 liveness
 *   GET  /api/model                  config, tokenizer info, tensor table with stats + histograms
 *   GET  /api/weights/:name          ?format=json (default, downsampled for heatmaps; &maxRows&maxCols&mode=mean|absmax)
 *                                    ?format=bin  (raw little-endian Float32, shape in X-Shape header)
 *   GET  /api/vocab                  {vocab: string[]} decoded text of every token id
 *   GET  /api/notebook               training logs (models/<run>/log.jsonl), results/*.json, model card
 *   POST /api/tokenize {code, cursor?}
 *   POST /api/complete {code, cursor?, maxTokens?, topK?, temperature?, stopAtNewline?}
 *   WS   /api/trace                  send {type:"trace", code, cursor?, maxTokens?, ...}; receives
 *                                    tokens -> trace -> step* -> done (or cancelled). Debounced per
 *                                    connection; a newer request cancels any older one.
 *
 * Everything else is served from web/dist (the built lab), else web/, else a placeholder page.
 * Binds to 127.0.0.1 only: the network lives on this PC.
 */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize, resolve, sep } from "node:path";
import type { AddressInfo } from "node:net";
import { WebSocketServer, type WebSocket } from "ws";
import { ModelHost, type HostOptions } from "./model-host.js";
import { buildPrompt, generate, tokenText, tracePrompt, type GenerateOptions, type TraceOptions } from "./inference.js";
import { readNotebook } from "./notebook.js";

export interface ServerOptions extends HostOptions {
  host?: string;
  port?: number;
  webDir?: string;
  /** Default debounce for WS trace requests (ms). */
  debounceMs?: number;
  quiet?: boolean;
}

export interface RunningServer {
  url: string;
  port: number;
  host: ModelHost;
  server: Server;
  close(): Promise<void>;
}

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml",
  ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon", ".woff2": "font/woff2", ".map": "application/json",
  ".wasm": "application/wasm", ".txt": "text/plain; charset=utf-8",
};

class HttpError extends Error {
  constructor(readonly status: number, msg: string) { super(msg); }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const s = JSON.stringify(body);
  res.writeHead(status, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", "content-length": Buffer.byteLength(s) });
  res.end(s);
}

async function readBody(req: IncomingMessage, limit = 1 << 20): Promise<unknown> {
  const chunks: Buffer[] = [];
  let n = 0;
  for await (const c of req) {
    n += (c as Buffer).length;
    if (n > limit) throw new HttpError(413, "body too large");
    chunks.push(c as Buffer);
  }
  if (n === 0) return {};
  try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch { throw new HttpError(400, "invalid JSON body"); }
}

function asCodeRequest(body: unknown): { code: string; cursor?: number } & Record<string, unknown> {
  if (!body || typeof body !== "object") throw new HttpError(400, "expected a JSON object");
  const b = body as Record<string, unknown>;
  if (typeof b.code !== "string") throw new HttpError(400, "`code` must be a string");
  if (b.code.length > 200_000) throw new HttpError(413, "`code` too long");
  if (b.cursor !== undefined && typeof b.cursor !== "number") throw new HttpError(400, "`cursor` must be a number");
  return b as { code: string; cursor?: number };
}

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function genOpts(b: Record<string, unknown>): GenerateOptions {
  return {
    maxTokens: num(b.maxTokens), topK: num(b.topK), temperature: num(b.temperature), seed: num(b.seed),
    stopAtNewline: typeof b.stopAtNewline === "boolean" ? b.stopAtNewline : undefined,
  };
}

function traceOpts(b: Record<string, unknown>): TraceOptions {
  return {
    topK: num(b.topK), attnWindow: num(b.attnWindow), mlpTopK: num(b.mlpTopK),
    lastVectors: typeof b.lastVectors === "boolean" ? b.lastVectors : undefined,
    mlpLastAll: typeof b.mlpLastAll === "boolean" ? b.mlpLastAll : undefined,
    mlpWindowTopK: num(b.mlpWindowTopK),
  };
}

/** Block-pool a [R, C] matrix down to at most maxRows x maxCols. */
export function downsample(data: Float32Array, R: number, C: number, maxRows: number, maxCols: number, mode: "mean" | "absmax") {
  const ry = Math.max(1, Math.ceil(R / maxRows));
  const rx = Math.max(1, Math.ceil(C / maxCols));
  const rows = Math.ceil(R / ry);
  const cols = Math.ceil(C / rx);
  const out = new Array<number>(rows * cols);
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    let s = 0, cnt = 0, best = 0;
    for (let y = i * ry; y < Math.min(R, (i + 1) * ry); y++) for (let x = j * rx; x < Math.min(C, (j + 1) * rx); x++) {
      const v = data[y * C + x];
      s += v; cnt++;
      if (Math.abs(v) > Math.abs(best)) best = v;
    }
    const v = mode === "absmax" ? best : s / cnt;
    out[i * cols + j] = Math.round(v * 1e4) / 1e4;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  return { rows, cols, pool: [ry, rx], min: mn, max: mx, data: out };
}

function placeholderPage(host: ModelHost): string {
  const info = host.modelInfo();
  return `<!doctype html><meta charset="utf-8"><title>Java Autocomplete Lab - server</title>
<style>body{font:15px/1.5 system-ui,sans-serif;background:#0b0e14;color:#d8dee9;max-width:760px;margin:40px auto;padding:0 16px}code{color:#88c0d0}a{color:#81a1c1}</style>
<h1>Java Autocomplete Lab - inference server</h1>
<p>The web lab is not built yet (no <code>web/dist</code>). The API is live:</p>
<ul><li><a href="/api/model">GET /api/model</a></li><li><a href="/api/weights/tok_emb">GET /api/weights/tok_emb</a></li>
<li>POST /api/complete, POST /api/tokenize</li><li>WS /api/trace</li></ul>
<p>Model: <code>${info.sourceLabel}</code>, ${info.numParams.toLocaleString()} params.</p>
${info.warnings.map((w) => `<p style="color:#ebcb8b">${w.replace(/</g, "&lt;")}</p>`).join("")}`;
}

export async function startServer(opts: ServerOptions = {}): Promise<RunningServer> {
  const log = opts.quiet ? () => {} : (m: string) => console.log(m);
  const host = new ModelHost({ ...opts, log: opts.log ?? log });
  const root = resolve(opts.root ?? process.cwd());
  const webCandidates = opts.webDir ? [resolve(root, opts.webDir)] : [join(root, "web", "dist"), join(root, "web")];
  const webRoot = webCandidates.find((d) => existsSync(join(d, "index.html")));
  const modelInfoJson = JSON.stringify(host.modelInfo());
  const debounceDefault = opts.debounceMs ?? 60;
  let vocabJson: string | null = null;

  async function handleApi(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
    const p = url.pathname;
    if (p === "/api/health" && req.method === "GET") return sendJson(res, 200, { ok: true, model: host.modelInfo().sourceLabel });
    if (p === "/api/model" && req.method === "GET") {
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(modelInfoJson);
      return;
    }
    if (p.startsWith("/api/weights/") && req.method === "GET") {
      const name = decodeURIComponent(p.slice("/api/weights/".length));
      const info = host.tensors.find((t) => t.name === name);
      if (!info) throw new HttpError(404, `no tensor named '${name}'`);
      const data = host.model.store.data.subarray(info.offset, info.offset + info.numel);
      const format = url.searchParams.get("format") ?? "json";
      if (format === "bin") {
        const copy = Buffer.from(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength));
        res.writeHead(200, {
          "content-type": "application/octet-stream", "x-shape": info.shape.join(","), "x-dtype": "f32le",
          "access-control-expose-headers": "x-shape, x-dtype", "content-length": copy.length, "cache-control": "no-store",
        });
        res.end(copy);
        return;
      }
      if (format !== "json") throw new HttpError(400, "format must be json or bin");
      const R = info.shape.length >= 2 ? info.shape[0] : 1;
      const C = info.numel / R;
      const maxRows = Math.max(1, Math.min(4096, Number(url.searchParams.get("maxRows") ?? 128) || 128));
      const maxCols = Math.max(1, Math.min(4096, Number(url.searchParams.get("maxCols") ?? 128) || 128));
      const mode = url.searchParams.get("mode") === "absmax" ? "absmax" : "mean";
      const ds = downsample(data, R, C, maxRows, maxCols, mode);
      return sendJson(res, 200, { name, shape: info.shape, mode, stats: host.statsByName.get(name), ...ds });
    }
    if (p === "/api/vocab" && req.method === "GET") {
      vocabJson ??= JSON.stringify({ vocab: Array.from({ length: host.tokenizer.vocabSize }, (_, i) => tokenText(host, i)) });
      res.writeHead(200, { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" });
      res.end(vocabJson);
      return;
    }
    if (p === "/api/notebook" && req.method === "GET") {
      return sendJson(res, 200, readNotebook(root, host));
    }
    if (p === "/api/tokenize" && req.method === "POST") {
      const b = asCodeRequest(await readBody(req));
      const pr = buildPrompt(host, b.code, b.cursor);
      return sendJson(res, 200, { tokens: pr.tokens, truncated: pr.truncated, prefixChars: pr.prefixChars });
    }
    if (p === "/api/complete" && req.method === "POST") {
      const b = asCodeRequest(await readBody(req));
      const t0 = performance.now();
      const pr = buildPrompt(host, b.code, b.cursor);
      // Stop generating as soon as the client gives up (the lab aborts stale requests on keystrokes).
      let gone = false;
      res.on("close", () => { if (!res.writableEnded) gone = true; });
      const g = await generate(host, pr.ids, genOpts(b), undefined, () => gone);
      if (g.cancelled) { if (!res.destroyed) sendJson(res, 499, { error: "cancelled" }); return; }
      return sendJson(res, 200, {
        completion: g.text, stopReason: g.stopReason,
        prompt: { tokens: pr.tokens, truncated: pr.truncated, prefixChars: pr.prefixChars },
        steps: g.steps, ms: Math.round(performance.now() - t0), model: host.modelInfo().sourceLabel,
      });
    }
    throw new HttpError(p.startsWith("/api/") ? 404 : 405, `no route ${req.method} ${p}`);
  }

  function serveStatic(res: ServerResponse, url: URL): void {
    if (!webRoot) {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(placeholderPage(host));
      return;
    }
    let rel = decodeURIComponent(url.pathname);
    if (rel.endsWith("/")) rel += "index.html";
    const file = normalize(join(webRoot, rel));
    if (!file.startsWith(webRoot + sep) && file !== webRoot) {
      res.writeHead(403); res.end("forbidden"); return;
    }
    let target = file;
    if (!existsSync(target) || statSync(target).isDirectory()) target = join(webRoot, "index.html"); // SPA fallback
    res.writeHead(200, { "content-type": MIME[extname(target).toLowerCase()] ?? "application/octet-stream" });
    res.end(readFileSync(target));
  }

  const server = createServer((req, res) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (url.pathname.startsWith("/api/")) {
      handleApi(req, res, url).catch((e: unknown) => {
        const status = e instanceof HttpError ? e.status : 500;
        if (status === 500) console.error(e);
        if (!res.headersSent) sendJson(res, status, { error: (e as Error).message });
        else res.end();
      });
      return;
    }
    if (req.method !== "GET" && req.method !== "HEAD") { res.writeHead(405); res.end(); return; }
    serveStatic(res, url);
  });

  const wss = new WebSocketServer({ noServer: true, maxPayload: 1 << 20 });
  server.on("upgrade", (req, socket, head) => {
    const url = new URL(req.url ?? "/", "http://127.0.0.1");
    if (url.pathname !== "/api/trace") { socket.destroy(); return; }
    wss.handleUpgrade(req, socket, head, (ws) => attachTrace(ws));
  });

  function attachTrace(ws: WebSocket): void {
    let latest = 0;
    let timer: NodeJS.Timeout | null = null;
    const send = (obj: unknown) => { if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj)); };
    send({ type: "hello", model: host.modelInfo().sourceLabel, config: host.model.cfg, warnings: host.warnings, debounceMs: debounceDefault });

    async function run(seq: number, reqId: unknown, b: Record<string, unknown>): Promise<void> {
      const stale = () => seq !== latest || ws.readyState !== ws.OPEN;
      if (stale()) return;
      const t0 = performance.now();
      const pr = buildPrompt(host, b.code as string, num(b.cursor));
      send({ type: "tokens", reqId, tokens: pr.tokens, truncated: pr.truncated, prefixChars: pr.prefixChars });
      const trace = tracePrompt(host, pr.ids, traceOpts(b));
      const traceMs = performance.now() - t0;
      if (stale()) { send({ type: "cancelled", reqId }); return; }
      send({ type: "trace", reqId, ms: Math.round(traceMs), ...trace });
      if (num(b.maxTokens) === 0) {
        send({ type: "done", reqId, completion: "", stopReason: "maxTokens", steps: 0, traceMs: Math.round(traceMs), totalMs: Math.round(performance.now() - t0) });
        return;
      }
      const g = await generate(host, pr.ids, genOpts(b), (s) => {
        if (!stale()) send({ type: "step", reqId, ...s, ms: Math.round(performance.now() - t0) });
      }, stale);
      if (g.cancelled) { send({ type: "cancelled", reqId }); return; }
      send({
        type: "done", reqId, completion: g.text, stopReason: g.stopReason, steps: g.steps.length,
        traceMs: Math.round(traceMs), totalMs: Math.round(performance.now() - t0),
      });
    }

    ws.on("message", (raw) => {
      let msg: Record<string, unknown>;
      try { msg = JSON.parse(raw.toString()) as Record<string, unknown>; } catch { send({ type: "error", error: "invalid JSON" }); return; }
      if (msg.type === "cancel") {
        latest++;
        if (timer) { clearTimeout(timer); timer = null; }
        send({ type: "cancelled", reqId: msg.reqId ?? null });
        return;
      }
      if (msg.type !== "trace") { send({ type: "error", error: "expected {type:'trace'} or {type:'cancel'}" }); return; }
      if (typeof msg.code !== "string" || msg.code.length > 200_000) { send({ type: "error", reqId: msg.reqId, error: "`code` must be a string" }); return; }
      const seq = ++latest;
      const reqId = msg.reqId ?? seq;
      const debounce = Math.max(0, Math.min(2000, num(msg.debounceMs) ?? debounceDefault));
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = null;
        run(seq, reqId, msg).catch((e: unknown) => send({ type: "error", reqId, error: (e as Error).message }));
      }, debounce);
    });
    ws.on("close", () => { latest++; if (timer) clearTimeout(timer); });
  }

  const port = opts.port ?? 5177;
  const bind = opts.host ?? "127.0.0.1";
  await new Promise<void>((res, rej) => {
    server.once("error", rej);
    server.listen(port, bind, () => { server.off("error", rej); res(); });
  });
  const actual = (server.address() as AddressInfo).port;
  const url = `http://${bind}:${actual}`;
  log(`[server] Java Autocomplete Lab listening on ${url}  (web: ${webRoot ?? "placeholder page"})`);
  return {
    url, port: actual, host, server,
    close: () => new Promise<void>((res) => {
      for (const c of wss.clients) c.terminate();
      wss.close();
      server.closeAllConnections?.();
      server.close(() => res());
    }),
  };
}
