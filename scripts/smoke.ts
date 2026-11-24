/**
 * `npm run smoke`: boot the inference server on a free port and exercise every endpoint.
 *
 *   GET /api/model, GET /api/weights/:name (json + bin), POST /api/complete, WS /api/trace
 *   (full trace, debounce of a burst of keystrokes, cancellation of a stale generation).
 *
 * Asserts shapes and real-time latency, prints measured timings and payload sizes, then
 * 'SMOKE OK' and exits 0. Any failure exits 1.
 */
import assert from "node:assert/strict";
import WebSocket from "ws";
import { startServer } from "../src/server/http.js";

const JAVA_PREFIX = `public class Main {
    public static void main(String[] args) {`;

type Msg = Record<string, any>;

function openWs(url: string): Promise<{ ws: WebSocket; msgs: Msg[]; bytes: Map<string, number>; waitFor: (pred: (m: Msg) => boolean, ms?: number) => Promise<Msg> }> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    const msgs: Msg[] = [];
    const bytes = new Map<string, number>();
    const waiters: { pred: (m: Msg) => boolean; res: (m: Msg) => void }[] = [];
    ws.on("message", (raw) => {
      const s = raw.toString();
      const m = JSON.parse(s) as Msg;
      bytes.set(`${m.reqId ?? ""}:${m.type}`, (bytes.get(`${m.reqId ?? ""}:${m.type}`) ?? 0) + s.length);
      msgs.push(m);
      for (let i = waiters.length - 1; i >= 0; i--) if (waiters[i].pred(m)) { waiters[i].res(m); waiters.splice(i, 1); }
    });
    ws.once("error", reject);
    ws.once("open", () => resolve({
      ws, msgs, bytes,
      waitFor: (pred, ms = 30_000) => new Promise<Msg>((res, rej) => {
        const hit = msgs.find(pred);
        if (hit) return res(hit);
        const t = setTimeout(() => rej(new Error("timeout waiting for WS message")), ms);
        waiters.push({ pred, res: (m) => { clearTimeout(t); res(m); } });
      }),
    }));
  });
}

async function main(): Promise<void> {
  const t0 = performance.now();
  const srv = await startServer({ port: 0, quiet: true });
  console.log(`server up at ${srv.url} in ${Math.round(performance.now() - t0)} ms; model: ${srv.host.modelInfo().sourceLabel}`);
  for (const w of srv.host.warnings) console.log(`  warning: ${w}`);
  try {
    // ---- /api/model
    const model = await (await fetch(`${srv.url}/api/model`)).json() as Msg;
    const cfg = model.config;
    for (const k of ["vocabSize", "ctx", "dModel", "nLayers", "nHeads", "dMlp"]) assert.ok(Number.isInteger(cfg[k]) && cfg[k] > 0, `config.${k}`);
    assert.equal(model.tokenizer.vocabSize, cfg.vocabSize);
    assert.equal(model.tensors.length, 2 + 12 * cfg.nLayers + 2, "tensor count");
    let total = 0;
    for (const t of model.tensors) {
      assert.equal(t.numel, t.shape.reduce((a: number, b: number) => a * b, 1), `${t.name} numel`);
      assert.equal(t.hist.length, 32);
      assert.ok(Number.isFinite(t.std) && Number.isFinite(t.mean));
      total += t.numel;
    }
    assert.equal(total, model.numParams);
    const tok = model.tensors.find((t: Msg) => t.name === "tok_emb");
    assert.deepEqual(tok.shape, [cfg.vocabSize, cfg.dModel]);
    console.log(`/api/model: ${model.numParams.toLocaleString()} params, ${model.tensors.length} tensors, d=${cfg.dModel} L=${cfg.nLayers} H=${cfg.nHeads} ctx=${cfg.ctx}, PCA explained ${model.pca.explained.map((e: number) => (e * 100).toFixed(1) + "%").join("/")}`);

    // ---- /api/weights
    const wj = await (await fetch(`${srv.url}/api/weights/tok_emb?maxRows=64&maxCols=32`)).json() as Msg;
    assert.ok(wj.rows <= 64 && wj.cols <= 32 && wj.data.length === wj.rows * wj.cols, "downsampled heatmap shape");
    const qkvName = "blocks.0.attn.qkv.W";
    const wbRes = await fetch(`${srv.url}/api/weights/${encodeURIComponent(qkvName)}?format=bin`);
    assert.equal(wbRes.status, 200);
    assert.equal(wbRes.headers.get("x-shape"), `${3 * cfg.dModel},${cfg.dModel}`);
    const wb = new Float32Array(await wbRes.arrayBuffer());
    assert.equal(wb.length, 3 * cfg.dModel * cfg.dModel);
    const ref = srv.host.model.store.get(qkvName).data;
    assert.ok(wb[0] === ref[0] && wb[wb.length - 1] === ref[ref.length - 1], "binary weights match the model");
    assert.equal((await fetch(`${srv.url}/api/weights/nope`)).status, 404);
    console.log(`/api/weights: heatmap ${wj.rows}x${wj.cols} (pool ${wj.pool.join("x")}), binary ${wb.length} floats`);

    // ---- /api/complete
    const code = JAVA_PREFIX;
    const cres = await fetch(`${srv.url}/api/complete`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ code, cursor: code.length, maxTokens: 16, stopAtNewline: false }),
    });
    assert.equal(cres.status, 200);
    const comp = await cres.json() as Msg;
    assert.equal(typeof comp.completion, "string");
    assert.ok(comp.steps.length >= 1 && comp.steps.length <= 16);
    for (const s of comp.steps) {
      assert.equal(s.topk.length, 10);
      assert.ok(s.topk[0].p >= s.topk[9].p && s.topk[0].p <= 1);
      assert.ok(Number.isInteger(s.id) && s.id >= 0 && s.id < cfg.vocabSize);
    }
    assert.equal(comp.prompt.tokens[0].id, 0, "prompt starts with <|bof|>");
    const msPerTok = comp.ms / comp.steps.length;
    console.log(`/api/complete: ${comp.prompt.tokens.length} prompt tokens -> ${comp.steps.length} tokens in ${comp.ms} ms (${msPerTok.toFixed(1)} ms/token): ${JSON.stringify(comp.completion)}`);
    const bad = await fetch(`${srv.url}/api/complete`, { method: "POST", body: "{\"code\": 3}" });
    assert.equal(bad.status, 400);

    // ---- WS /api/trace: one full trace
    const wsUrl = srv.url.replace("http", "ws") + "/api/trace";
    const c = await openWs(wsUrl);
    const hello = await c.waitFor((m) => m.type === "hello");
    assert.equal(hello.config.dModel, cfg.dModel);
    const sentAt = performance.now();
    c.ws.send(JSON.stringify({ type: "trace", reqId: "a", code, cursor: code.length, maxTokens: 8, stopAtNewline: false, debounceMs: 0 }));
    const toks = await c.waitFor((m) => m.type === "tokens" && m.reqId === "a");
    const tr = await c.waitFor((m) => m.type === "trace" && m.reqId === "a");
    const traceLatency = performance.now() - sentAt;
    const done = await c.waitFor((m) => m.type === "done" && m.reqId === "a");
    const totalLatency = performance.now() - sentAt;
    const T = toks.tokens.length;
    const S = 1 + 2 * cfg.nLayers;
    assert.equal(tr.T, T);
    assert.equal(tr.stages.length, S);
    assert.equal(tr.pca.length, S);
    for (const p of tr.pca) assert.equal(p.length, T * 3, "PCA [T*3] per stage");
    for (const n of tr.norms) assert.equal(n.length, T);
    assert.equal(tr.lastVectors.length, S);
    for (const v of tr.lastVectors) assert.equal(v.length, cfg.dModel);
    assert.equal(tr.attention.last.length, cfg.nLayers);
    for (const l of tr.attention.last) {
      assert.equal(l.length, cfg.nHeads);
      for (const row of l) {
        assert.equal(row.length, T);
        const sum = row.reduce((a: number, b: number) => a + b, 0);
        assert.ok(Math.abs(sum - 1) < 0.02, `attention row sums to 1 (got ${sum})`);
      }
    }
    const W = tr.attention.window;
    assert.equal(tr.attention.windowStart, T - W);
    for (const l of tr.attention.matrices) for (const m of l) {
      assert.equal(m.length, W * W);
      for (let q = 0; q < W; q++) for (let k = q + 1; k < W; k++) assert.equal(m[q * W + k], 0, "causal mask");
    }
    assert.equal(tr.mlp.lastTop.length, cfg.nLayers);
    for (const l of tr.mlp.lastTop) for (const n of l) assert.ok(n.i >= 0 && n.i < cfg.dMlp);
    for (const am of tr.mlp.argmax) assert.equal(am.length, T);
    assert.equal(tr.next.topk.length, 10);
    assert.equal(tr.logitLens.length, S);
    for (const l of tr.logitLens) assert.equal(l.topk.length, 5);
    // the logit lens at the last stage is exactly the model's own prediction
    assert.equal(tr.logitLens[S - 1].topk[0].id, tr.next.topk[0].id, "logit lens of the final stage == model output");
    const steps = c.msgs.filter((m) => m.type === "step" && m.reqId === "a");
    assert.equal(steps.length, done.steps);
    assert.ok(steps.length >= 1);
    for (const s of steps) assert.equal(s.topk.length, 10);
    const traceBytes = c.bytes.get("a:trace") ?? 0;
    const stepBytes = c.bytes.get("a:step") ?? 0;
    console.log(`WS trace: ${T} tokens, ${S} residual stages, attn window ${W}; trace frame ${(traceBytes / 1024).toFixed(1)} KB (server ${tr.ms} ms, round-trip ${traceLatency.toFixed(0)} ms); ${steps.length} steps ${(stepBytes / 1024).toFixed(1)} KB; done in ${totalLatency.toFixed(0)} ms`);
    assert.ok(traceLatency < 1000, `trace must arrive well under 1 s (got ${traceLatency.toFixed(0)} ms)`);

    // ---- WS: worst case, a prefix longer than the context window (T = ctx)
    const longCode = Array.from({ length: 60 }, (_, i) => `        int value${i} = compute(${i}, args.length) + offset;\n`).join("");
    const longPrefix = `${JAVA_PREFIX}\n${longCode}        System.out.println(`;
    const sentFull = performance.now();
    c.ws.send(JSON.stringify({ type: "trace", reqId: "full", code: longPrefix, maxTokens: 16, stopAtNewline: false, debounceMs: 0 }));
    const fullTok = await c.waitFor((m) => m.type === "tokens" && m.reqId === "full");
    const fullTr = await c.waitFor((m) => m.type === "trace" && m.reqId === "full");
    const fullTraceMs = performance.now() - sentFull;
    const fullDone = await c.waitFor((m) => m.type === "done" && m.reqId === "full");
    const fullTotalMs = performance.now() - sentFull;
    assert.equal(fullTr.T, cfg.ctx, "window cropped to ctx");
    assert.ok(fullTok.truncated > 0);
    const fullBytes = c.bytes.get("full:trace") ?? 0;
    console.log(`WS full-context trace: T=${fullTr.T} (${fullTok.truncated} tokens cropped), trace frame ${(fullBytes / 1024).toFixed(1)} KB, trace round-trip ${fullTraceMs.toFixed(0)} ms, ${fullDone.steps} tokens done in ${fullTotalMs.toFixed(0)} ms`);
    assert.ok(fullTraceMs < 1000, `full-context trace must arrive well under 1 s (got ${fullTraceMs.toFixed(0)} ms)`);

    // ---- WS: a burst of keystrokes is debounced to one run
    const burst = ["public static", "public static void", "public static void ma", "public static void main("];
    burst.forEach((s, i) => c.ws.send(JSON.stringify({ type: "trace", reqId: `b${i}`, code: s, maxTokens: 4, debounceMs: 80 })));
    await c.waitFor((m) => m.type === "done" && m.reqId === "b3");
    for (let i = 0; i < 3; i++) assert.ok(!c.msgs.some((m) => m.reqId === `b${i}` && (m.type === "tokens" || m.type === "trace")), `debounced request b${i} must not run`);
    console.log(`WS debounce: 4 keystrokes within 80 ms -> 1 traced run (b3)`);

    // ---- WS: a new request cancels a running generation
    c.ws.send(JSON.stringify({ type: "trace", reqId: "long", code: JAVA_PREFIX, maxTokens: 128, stopAtNewline: false, debounceMs: 0 }));
    await c.waitFor((m) => m.type === "step" && m.reqId === "long");
    c.ws.send(JSON.stringify({ type: "trace", reqId: "new", code: JAVA_PREFIX + "\n        int", maxTokens: 4, debounceMs: 0 }));
    await c.waitFor((m) => m.type === "done" && m.reqId === "new");
    const longDone = c.msgs.find((m) => m.type === "done" && m.reqId === "long");
    const longCancelled = c.msgs.find((m) => m.type === "cancelled" && m.reqId === "long");
    const longSteps = c.msgs.filter((m) => m.type === "step" && m.reqId === "long").length;
    assert.ok(longCancelled && !longDone, "stale generation must be cancelled");
    assert.ok(longSteps < 128);
    console.log(`WS cancel: stale 128-token generation stopped after ${longSteps} steps when a newer prefix arrived`);
    c.ws.close();
  } finally {
    await srv.close();
  }
  console.log(`SMOKE OK (${Math.round(performance.now() - t0)} ms)`);
}

main().then(() => process.exit(0), (e: unknown) => {
  console.error("SMOKE FAILED:", e);
  process.exit(1);
});
