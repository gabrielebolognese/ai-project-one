/**
 * `npm run screenshots`: start the inference server, drive the built lab in headless Chromium,
 * type real Java, and save PNGs to docs/screenshots/. Also records the measured frame rate and
 * trace/cancel counters while typing to results/web-perf.json.
 *
 * Requires `npm run build` first (it serves web/dist). Options:
 *   --model models/<run>   serve a specific checkpoint (default: the server's resolution order)
 *   --out docs/screenshots
 */
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { spawn } from "node:child_process";
import { createServer, type AddressInfo } from "node:net";
import { chromium, type Page } from "playwright";

const { values: a } = parseArgs({
  options: {
    model: { type: "string" },
    out: { type: "string", default: join("docs", "screenshots") },
    "type-delay": { type: "string", default: "55" },
  },
});

if (!existsSync(join("web", "dist", "index.html"))) {
  console.error("web/dist is missing: run `npm run build` first.");
  process.exit(2);
}
mkdirSync(a.out!, { recursive: true });

// Typed after the starter snippet (which ends with `return items.`). Newlines rely on the
// editor's auto-indent, exactly like a person typing.
const TYPED = `size();
}

public boolean contains(String item) {
return items.contains(item);
}

public String get(int index) {
if (index < 0 || index >= items.size()) {
throw new IndexOutOfBoundsException("index: " + index);
}
return items.`;

type Lab = {
  setTab(t: string): void;
  setMode(m: string): void;
  net: { setLayer(l: number): void; playPass(): void; stopReplay(): void };
  weights: { select(name: string, lens?: string): Promise<void>; zoomTo(f: number): void };
  stats(): { fps: number; fpsSamples: number[]; trace: Record<string, number>; completions: number; lastComplete: string | null; hasTrace: boolean };
  focusEditor(): void;
};

const lab = <T>(page: Page, fn: (l: Lab, arg: unknown) => T, arg?: unknown) =>
  page.evaluate(([src, x]) => new Function("l", "x", `return (${src})(l, x)`)((window as unknown as { __lab: Lab }).__lab, x), [fn.toString(), arg] as const) as Promise<Awaited<T>>;

async function waitFor(page: Page, pred: (l: Lab) => boolean, timeout = 30000): Promise<void> {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) {
    if (await lab(page, pred).catch(() => false)) return;
    await page.waitForTimeout(150);
  }
  throw new Error(`timed out waiting for ${pred.toString()}`);
}

// The server runs as its own process (exactly like `npm start`), so its CPU work does not stall
// the Playwright driver and the measured typing is realistic.
const port = await new Promise<number>((res) => {
  const s = createServer();
  s.listen(0, "127.0.0.1", () => { const p = (s.address() as AddressInfo).port; s.close(() => res(p)); });
});
const child = spawn(process.execPath, ["--import", "tsx", join("src", "server", "main.ts"), "--port", String(port), ...(a.model ? ["--model", a.model] : [])], {
  stdio: ["ignore", "pipe", "inherit"],
});
await new Promise<void>((res, rej) => {
  const timer = setTimeout(() => rej(new Error("server did not start within 180 s")), 180_000);
  child.stdout!.on("data", (d: Buffer) => { if (String(d).includes("listening")) { clearTimeout(timer); res(); } });
  child.on("exit", (c) => rej(new Error(`server exited with code ${c}`)));
});
const server = { url: `http://127.0.0.1:${port}` };
const health = (await (await fetch(`${server.url}/api/health`)).json()) as { model: string };
console.log(`[screenshots] server ${server.url} | model ${health.model}`);
// GPU raster through ANGLE/D3D11, like desktop Chrome or Edge on this laptop. Falls back to
// headless software rendering (SwiftShader) if that fails; web-perf.json records which one ran.
const GPU_ARGS = ["--enable-gpu", "--use-angle=d3d11", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"];
const browser = await chromium.launch({ args: process.argv.includes("--software") ? [] : GPU_ARGS }).catch(() => chromium.launch());
const shots: string[] = [];
const shot = async (page: Page, name: string) => {
  const p = join(a.out!, name);
  await page.screenshot({ path: p });
  shots.push(p);
  console.log(`[screenshots] ${p}`);
};

try {
  // ---- 1920 x 1080 --------------------------------------------------------------------------
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  page.on("pageerror", (e) => console.error("[page error]", e.message));
  await page.goto(server.url);
  await waitFor(page, (l) => l.stats().hasTrace);
  await page.click(".cm-content");
  await page.keyboard.press("Control+End");
  const before = await lab(page, (l) => l.stats());
  const t0 = Date.now();
  await page.keyboard.type(TYPED, { delay: Number(a["type-delay"]) });
  const typingMs = Date.now() - t0;
  const during = await lab(page, (l) => l.stats());
  await waitFor(page, (l) => l.stats().completions > 0 && l.stats().hasTrace);
  await page.waitForTimeout(1800);
  const after = await lab(page, (l) => l.stats());
  await shot(page, "01-lab-network.png");

  // attention close-up of layer 1 and layer 3
  await page.selectOption("#layer", "0");
  await lab(page, (l) => l.setMode("heads"));
  await page.waitForTimeout(900);
  await shot(page, "02-attention-closeup.png");
  await page.selectOption("#layer", "2");
  await page.waitForTimeout(900);
  await shot(page, "02b-attention-layer3.png");
  await page.selectOption("#layer", "-1");
  await lab(page, (l) => l.setMode("tower"));

  // replay of the forward pass, caught half-way up the tower
  await page.click("#play");
  await page.waitForTimeout(900 * 5 + 350);
  await shot(page, "05-replay-midpass.png");
  await lab(page, (l) => l.net.stopReplay());

  // weights
  await lab(page, (l) => l.setTab("weights"));
  await lab(page, (l) => l.weights.select("blocks.0.attn.qkv.W"));
  await page.waitForTimeout(1100);
  await page.mouse.move(980, 520);
  await page.waitForTimeout(200);
  await shot(page, "03-weights-qkv.png");
  await lab(page, (l) => l.weights.select("pos_emb", "cosine"));
  await page.waitForTimeout(1000);
  await shot(page, "06-weights-posemb-cosine.png");
  await lab(page, (l) => l.weights.select("blocks.1.mlp.fc.W"));
  await page.waitForTimeout(900);
  await lab(page, (l) => l.weights.zoomTo(26));
  await page.mouse.move(1000, 560);
  await page.waitForTimeout(300);
  await shot(page, "07-weights-zoom-values.png");
  await lab(page, (l) => l.weights.select("tok_emb", "pca"));
  await page.waitForTimeout(1200);
  await shot(page, "08-weights-vocab-pca.png");

  // lab notebook
  await lab(page, (l) => l.setTab("notebook"));
  await page.waitForTimeout(1900);
  await shot(page, "04-lab-notebook.png");

  // ---- 1366 x 768 ---------------------------------------------------------------------------
  const small = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });
  small.on("pageerror", (e) => console.error("[page error]", e.message));
  await small.goto(server.url);
  await waitFor(small, (l) => l.stats().hasTrace);
  await small.click(".cm-content");
  await small.keyboard.press("Control+End");
  await small.keyboard.type("size();\n}\n\npublic void clear() {\nitems.", { delay: 40 });
  await waitFor(small, (l) => l.stats().completions > 0);
  await small.waitForTimeout(1600);
  await shot(small, "09-lab-1366x768.png");
  await lab(small, (l) => l.setTab("weights"));
  await lab(small, (l) => l.weights.select("blocks.3.mlp.proj.W"));
  await small.waitForTimeout(1000);
  await shot(small, "10-weights-1366x768.png");

  const renderer = await page.evaluate(`(() => { const gl = document.createElement("canvas").getContext("webgl"); const d = gl && gl.getExtension("WEBGL_debug_renderer_info"); return gl && d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : "unknown"; })()`);
  const samples = during.fpsSamples.slice(before.fpsSamples.length);
  const sorted = [...samples].sort((x, y) => x - y);
  const perf = {
    date: new Date().toISOString(),
    browser: `headless Chromium ${browser.version()}`,
    renderer,
    viewport: "1920x1080",
    model: health.model,
    typedChars: TYPED.length,
    typingSeconds: Math.round(typingMs / 100) / 10,
    fpsMedianWhileTyping: sorted.length ? Math.round(sorted[Math.floor(sorted.length / 2)]) : null,
    fpsMinWhileTyping: sorted.length ? Math.round(sorted[0]) : null,
    traceRequestsSent: after.trace.sent - before.trace.sent,
    tracesRendered: after.trace.traces - before.trace.traces,
    cancelledByServer: after.trace.cancelled - before.trace.cancelled,
    staleTracesDroppedByClient: after.trace.dropped - before.trace.dropped,
    supersededBeforeComputing: (after.trace.sent - before.trace.sent) - (after.trace.traces - before.trace.traces) - (after.trace.dropped - before.trace.dropped) - (after.trace.cancelled - before.trace.cancelled),
    lastTraceMs: after.trace.lastMs,
    completionsShown: after.completions,
    lastCompletion: after.lastComplete,
    note: "Measured by npm run screenshots (headless Chromium; see renderer) while other jobs were loading the laptop CPU. The server debounces (60 ms) and cancels superseded passes; the client renders any trace newer than the last one rendered, so it stays at most one forward pass behind the keyboard.",
  };
  writeFileSync(join("results", "web-perf.json"), JSON.stringify(perf, null, 2) + "\n");
  console.log(`[screenshots] perf ${JSON.stringify(perf)}`);
} finally {
  await browser.close();
  child.kill();
}
console.log(`[screenshots] saved ${shots.length} PNGs to ${a.out}`);
