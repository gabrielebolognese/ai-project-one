/**
 * Lab notebook: training curves from models/<run>/log.jsonl, the evaluation tables from
 * results/results.json (or an explicit "not run yet"), the benchmark that chose the model size,
 * the corpus, and a model card. Every number comes from files the scripts wrote (GET /api/notebook).
 */
import type { ModelInfo } from "../api";
import { LineChart } from "../charts";
import { C, fmt, fmtInt, pct } from "../theme";
import { TweenNum } from "../tween";

interface LogRow {
  t: number; rho: number; step: number; tokens: number; lr: number; ctx: number; loss: number; gnorm: number; tok_s: number;
  val_loss?: number; val_loss_raw?: number; best?: boolean;
}

interface Run {
  name: string;
  loaded: boolean;
  config: { model?: Record<string, number>; preset?: string; status?: string; best?: Record<string, number>; train?: { workers?: number } } | null;
  summary: Record<string, number | string> | null;
  log: LogRow[];
}

export interface Notebook {
  model: string;
  source: { kind: string; run?: string };
  runs: Run[];
  results: unknown;
  resultsMd: string | null;
  bench: {
    date?: string;
    machine?: { cpu?: string; logicalCores?: number };
    candidates?: { preset: string; params: number; tok_s: number; sustained_gflops: number; tokens_in_budget: number; tokens_per_param: number }[];
    scaling?: { workers: number; tok_s: number; sustained_gflops: number }[];
    choice?: { preset: string; reason: string };
    notes?: string[];
  } | null;
  corpus: { train_tokens?: number; val_tokens?: number; test_tokens?: number; files?: number; vocab_size?: number; repos?: Record<string, Record<string, { files: number; tokens: number }>> } | null;
  webPerf: Record<string, number | string> | null;
  modelCard: string | null;
}

export class NotebookView {
  private nb: Notebook | null = null;
  private info: ModelInfo | null = null;
  private runIdx = 0;
  private charts: LineChart[] = [];
  private appear = new TweenNum(0, 1400);

  constructor(private root: HTMLElement) {}

  set(nb: Notebook, info: ModelInfo): void {
    this.nb = nb;
    this.info = info;
    this.render();
  }

  show(): void {
    this.appear.jump(0);
    this.appear.set(1, performance.now(), 1400);
  }

  private render(): void {
    const nb = this.nb!, info = this.info!;
    const run = nb.runs[this.runIdx];
    this.root.innerHTML = `
      <div class="nb-left">
        <section class="nb-sec">
          <div class="nb-h"><h2>Training curves</h2>
            <div class="nb-runs">${nb.runs.map((r, i) => `<button class="chip${i === this.runIdx ? " on" : ""}" data-run="${i}">${r.name}${r.loaded ? " · loaded" : ""}</button>`).join("") || `<span class="muted">no training runs in models/ yet</span>`}</div></div>
          ${run ? runSummary(run) : ""}
          <div class="nb-charts">
            <canvas class="nb-chart"></canvas><canvas class="nb-chart"></canvas><canvas class="nb-chart"></canvas><canvas class="nb-chart"></canvas>
          </div>
        </section>
        <section class="nb-sec"><div class="nb-h"><h2>Evaluation</h2><span class="muted">results/results.json</span></div>${renderResults(nb)}</section>
        <section class="nb-sec"><div class="nb-h"><h2>Model-size benchmark</h2><span class="muted">results/bench.json${nb.bench?.date ? ` · ${nb.bench.date.slice(0, 10)}` : ""}</span></div>${renderBench(nb)}</section>
        <section class="nb-sec"><div class="nb-h"><h2>Corpus</h2><span class="muted">results/corpus-stats.json</span></div>${renderCorpus(nb)}</section>
      </div>
      <aside class="nb-right"><section class="nb-sec card">${nb.modelCard ? md(nb.modelCard) : modelCard(nb, info)}</section></aside>`;
    this.root.querySelectorAll<HTMLButtonElement>("[data-run]").forEach((b) => b.addEventListener("click", () => { this.runIdx = Number(b.dataset.run); this.render(); this.show(); }));
    const cv = [...this.root.querySelectorAll<HTMLCanvasElement>(".nb-chart")];
    const log = run?.log ?? [];
    const val = log.filter((r) => r.val_loss !== undefined);
    const tokM = (r: LogRow) => r.tokens / 1e6;
    this.charts = [
      new LineChart(cv[0], [
        { label: "train loss", color: C.blue, points: log.map((r) => [tokM(r), r.loss]), width: 1.3 },
        { label: "val (EMA weights)", color: C.yellow, points: val.map((r) => [tokM(r), r.val_loss!]), dots: true, width: 2 },
        { label: "val (raw)", color: C.teal, points: val.filter((r) => r.val_loss_raw !== undefined).map((r) => [tokM(r), r.val_loss_raw!]), dots: true, dash: [4, 3] },
      ], { title: "Cross-entropy (nats/token)", xLabel: "tokens seen (M)", xFmt: (v) => fmt(v, 1), yFmt: (v) => fmt(v, 2), note: "no log.jsonl yet" }),
      new LineChart(cv[1], [
        { label: "learning rate", color: C.yellow, points: log.map((r) => [r.rho, r.lr]) },
      ], { title: "Budget-aware schedule: LR vs progress ρ = elapsed / budget", xLabel: "ρ", xFmt: (v) => fmt(v, 2), yFmt: (v) => v.toExponential(1) }),
      new LineChart(cv[2], [
        { label: "tokens / s", color: C.teal, points: log.map((r) => [r.t / 60, r.tok_s]) },
      ], { title: `Throughput on the Ryzen 7 5700U (${run?.config?.train?.workers ?? "?"} data-parallel workers)`, xLabel: "minutes", xFmt: (v) => fmt(v, 0), yFmt: (v) => fmt(v, 0) }),
      new LineChart(cv[3], [
        { label: "grad norm", color: C.red, points: log.map((r) => [r.step, r.gnorm]) },
        { label: "context", color: C.purple, points: log.map((r) => [r.step, r.ctx / 100]), dash: [3, 3] },
      ], { title: "Gradient norm (clip 1.0) · context curriculum (÷100)", xLabel: "step", xFmt: (v) => fmt(v, 0), yFmt: (v) => fmt(v, 2) }),
    ];
  }

  frame(now: number): void {
    const p = this.appear.get(now);
    for (const c of this.charts) c.draw(p);
  }
}

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function runSummary(run: Run): string {
  const s = run.summary;
  const last = run.log[run.log.length - 1];
  const vals = run.log.filter((r) => r.val_loss !== undefined);
  const best = vals.reduce<LogRow | null>((b, r) => (!b || r.val_loss! < b.val_loss! ? r : b), null);
  const cells: [string, string][] = [];
  const m = run.config?.model;
  if (m) cells.push(["model", `${m.nLayers}L · d${m.dModel} · ${m.nHeads}H · ctx ${m.ctx}`]);
  if (s) {
    cells.push(["status", String(s.completed) === "false" || run.config?.status === "stopped-early" ? "stopped early" : "finished"]);
    cells.push(["steps", fmtInt(Number(s.steps))], ["tokens", `${(Number(s.tokens) / 1e6).toFixed(2)} M`], ["wall clock", `${(Number(s.elapsedSec) / 60).toFixed(1)} min`]);
    cells.push(["best val loss", fmt(Number(s.bestValLoss), 4)], ["throughput", `${fmtInt(Number(s.tokensPerSec))} tok/s`], ["sustained", `${fmt(Number(s.sustainedGflops), 1)} GFLOPS`]);
  } else if (last) {
    cells.push(["status", `in progress (ρ = ${fmt(last.rho, 2)})`], ["steps", fmtInt(last.step)], ["tokens", `${(last.tokens / 1e6).toFixed(2)} M`], ["elapsed", `${(last.t / 60).toFixed(1)} min`]);
    if (best) cells.push(["best val loss", fmt(best.val_loss!, 4)]);
  }
  if (best) cells.push(["best val ppl", fmt(Math.exp(best.val_loss!), 1)]);
  return `<div class="kv">${cells.map(([k, v]) => `<div><span>${k}</span><b>${esc(v)}</b></div>`).join("")}</div>`;
}

function renderResults(nb: Notebook): string {
  if (!nb.results) {
    return `<div class="pending"><b>Not measured yet.</b> The held-out evaluation (<code>npm run evaluate</code>: top-1 / top-5 accuracy, perplexity,
      identifier accuracy and latency against the Kneser-Ney 4-gram and bigram baselines on jsoup, gson and retrofit) has not run on this checkout,
      so there is no <code>results/results.json</code>. Nothing is shown rather than invented numbers.</div>`;
  }
  return generic(nb.results, 0);
}

function generic(v: unknown, depth: number): string {
  if (v === null || v === undefined) return `<span class="muted">–</span>`;
  if (typeof v !== "object") return esc(typeof v === "number" ? fmtNum(v) : String(v));
  if (Array.isArray(v)) {
    if (v.length && v.every((x) => x && typeof x === "object" && !Array.isArray(x))) {
      const cols = [...new Set(v.flatMap((x) => Object.keys(x as object)))].filter((k) => (v as Record<string, unknown>[]).every((x) => typeof x[k] !== "object" || x[k] === null));
      return `<table class="tbl"><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${v.map((row) => `<tr>${cols.map((c) => `<td>${generic((row as Record<string, unknown>)[c], depth + 1)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    }
    return esc(v.map((x) => (typeof x === "number" ? fmtNum(x) : JSON.stringify(x))).join(", "));
  }
  const o = v as Record<string, unknown>;
  const scal = Object.entries(o).filter(([, x]) => x === null || typeof x !== "object" || (Array.isArray(x) && x.every((y) => typeof y !== "object")));
  const nested = Object.entries(o).filter(([, x]) => x !== null && typeof x === "object" && !(Array.isArray(x) && x.every((y) => typeof y !== "object")));
  let html = "";
  if (scal.length) html += `<table class="tbl kvt"><tbody>${scal.map(([k, x]) => `<tr><th>${esc(k)}</th><td>${generic(x, depth + 1)}</td></tr>`).join("")}</tbody></table>`;
  for (const [k, x] of nested) html += depth < 3 ? `<h4>${esc(k)}</h4>${generic(x, depth + 1)}` : `<pre>${esc(JSON.stringify(x, null, 1).slice(0, 600))}</pre>`;
  return html;
}

function fmtNum(v: number): string {
  if (Number.isInteger(v)) return fmtInt(v);
  return Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(2);
}

function renderBench(nb: Notebook): string {
  const b = nb.bench;
  if (!b?.candidates) return `<div class="pending">No benchmark yet.</div>`;
  const rows = b.candidates.map((c) => `<tr class="${b.choice?.preset === c.preset ? "hl" : ""}"><td>${c.preset}</td><td>${fmtInt(c.params)}</td><td>${fmt(c.tok_s, 0)}</td><td>${fmt(c.sustained_gflops, 1)}</td><td>${(c.tokens_in_budget / 1e6).toFixed(1)} M</td><td>${fmt(c.tokens_per_param, 1)}</td></tr>`).join("");
  const sc = b.scaling?.length ? `<table class="tbl small"><thead><tr><th>workers</th>${b.scaling.map((s) => `<th>${s.workers}</th>`).join("")}</tr></thead><tbody>
    <tr><td>tok/s</td>${b.scaling.map((s) => `<td>${fmt(s.tok_s, 0)}</td>`).join("")}</tr>
    <tr><td>GFLOPS</td>${b.scaling.map((s) => `<td>${fmt(s.sustained_gflops, 1)}</td>`).join("")}</tr></tbody></table>` : "";
  return `<table class="tbl"><thead><tr><th>preset</th><th>params</th><th>tok/s</th><th>GFLOPS</th><th>tokens in 150 min</th><th>tokens/param</th></tr></thead><tbody>${rows}</tbody></table>
    ${b.choice ? `<p class="note">Chosen: <b>${b.choice.preset}</b>. ${esc(b.choice.reason)}</p>` : ""}${sc}
    ${b.notes?.length ? `<ul class="notes">${b.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}`;
}

function renderCorpus(nb: Notebook): string {
  const c = nb.corpus;
  if (!c) return `<div class="pending">No corpus statistics yet.</div>`;
  const split = (name: string) => Object.entries(c.repos?.[name] ?? {}).sort((a, b) => b[1].tokens - a[1].tokens);
  const train = split("train");
  const held = [...split("val").map((x) => [...x, "val"] as const), ...split("test").map((x) => [...x, "test"] as const)];
  const tot = train.reduce((s, [, r]) => s + r.tokens, 0) || 1;
  return `<div class="kv"><div><span>files</span><b>${fmtInt(c.files ?? 0)}</b></div><div><span>train tokens</span><b>${fmtInt(c.train_tokens ?? 0)}</b></div>
    <div><span>val tokens</span><b>${fmtInt(c.val_tokens ?? 0)}</b></div><div><span>test tokens</span><b>${fmtInt(c.test_tokens ?? 0)}</b></div><div><span>BPE vocab</span><b>${fmtInt(c.vocab_size ?? 0)}</b></div></div>
    <div class="repo-bars">${train.slice(0, 14).map(([n, r]) => `<div class="rb"><span>${esc(n)}</span><i style="width:${(r.tokens / tot) * 100 * 3.2}%"></i><b>${(r.tokens / 1e6).toFixed(2)}M</b></div>`).join("")}</div>
    <p class="note">Held out by repository (never seen in training): ${held.map(([n, r, s]) => `<b>${esc(n)}</b> (${s}, ${(r.tokens / 1e6).toFixed(2)}M)`).join(", ")}.</p>`;
}

function modelCard(nb: Notebook, info: ModelInfo): string {
  const c = info.config;
  const run = nb.runs.find((r) => r.loaded);
  const s = run?.summary;
  const vals = run?.log.filter((r) => r.val_loss !== undefined) ?? [];
  const best = vals.reduce<LogRow | null>((b, r) => (!b || r.val_loss! < b.val_loss! ? r : b), null);
  const warn = info.source.kind === "random" ? `<p class="warn">The server is running a <b>randomly initialised</b> model (no checkpoint in models/). Everything you see is real computation, but the weights are untrained.</p>` : "";
  const preview = info.source.kind === "run" ? `<p class="warn">Loaded checkpoint: <b>${esc(info.sourceLabel)}</b>. This is a training run, not models/release.</p>` : "";
  return `<h2>Model card</h2>${warn}${preview}
  <h3>Model</h3>
  <table class="tbl kvt"><tbody>
    <tr><th>type</th><td>decoder-only transformer, pre-LayerNorm, GELU MLP, learned absolute positions, tied input/output embedding</td></tr>
    <tr><th>shape</th><td>${c.nLayers} layers · d<sub>model</sub> ${c.dModel} · ${c.nHeads} heads (d<sub>head</sub> ${c.dModel / c.nHeads}) · MLP ${c.dMlp} · context ${c.ctx}</td></tr>
    <tr><th>parameters</th><td>${fmtInt(info.numParams)} (fp32, ${(info.numParams * 4 / 1e6).toFixed(1)} MB)</td></tr>
    <tr><th>tokenizer</th><td>byte-level BPE, ${info.tokenizer.vocabSize} tokens (${fmtInt(info.tokenizer.numMerges)} merges), trained on the training split</td></tr>
    <tr><th>implementation</th><td>hand-written TypeScript autograd + WASM SIMD GEMM (${esc(info.kernel)}); no ML framework, no Python</td></tr>
  </tbody></table>
  <h3>Training</h3>
  <table class="tbl kvt"><tbody>
    <tr><th>hardware</th><td>${esc(nb.bench?.machine?.cpu?.trim() ?? "AMD Ryzen 7 5700U")}, CPU only</td></tr>
    <tr><th>objective</th><td>next-token cross-entropy, AdamW (β = 0.9, 0.95), wall-clock budget-aware cosine LR, EMA weights, context curriculum 64 → 128</td></tr>
    ${run ? `<tr><th>run</th><td>${esc(run.name)}${s ? `: ${fmtInt(Number(s.steps))} steps, ${(Number(s.tokens) / 1e6).toFixed(2)} M tokens, ${(Number(s.elapsedSec) / 60).toFixed(0)} min` : run.log.length ? ` (in progress, ${(run.log[run.log.length - 1].tokens / 1e6).toFixed(2)} M tokens)` : ""}</td></tr>` : ""}
    ${best ? `<tr><th>best val loss</th><td>${fmt(best.val_loss!, 4)} nats/token (perplexity ${fmt(Math.exp(best.val_loss!), 1)}) on commons-codec</td></tr>` : ""}
  </tbody></table>
  <h3>Data</h3>
  <p>${fmtInt(nb.corpus?.train_tokens ?? 0)} training tokens of Java from permissively licensed (Apache-2.0 / MIT) open-source repositories; see <code>data/SOURCES.md</code>. Validation and test splits are whole held-out repositories.</p>
  <h3>Intended use</h3>
  <p>An inspectable teaching and research model: watching a small transformer complete Java one token at a time, with every weight and activation visible. It runs entirely on the local PC.</p>
  <h3>Limitations</h3>
  <ul><li>About a million parameters and a ${c.ctx}-token window: it learns syntax, idioms and local identifier reuse, not program semantics.</li>
  <li>Suggestions can be syntactically plausible and wrong. It is not a coding assistant.</li>
  <li>Trained only on library-style Java from a few projects, so its style follows Apache Commons, Guava and similar code.</li></ul>
  ${nb.webPerf ? `<h3>Lab performance</h3><table class="tbl kvt"><tbody>${Object.entries(nb.webPerf).map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(typeof v === "number" ? fmtNum(v) : String(v))}</td></tr>`).join("")}</tbody></table>` : ""}
  <p class="muted">Licence: MIT (code). Corpus files keep their original Apache-2.0 / MIT licences.</p>`;
}

/** Minimal markdown for MODEL_CARD.md (headings, lists, tables, code, emphasis). */
function md(src: string): string {
  const lines = src.replace(/\r/g, "").split("\n");
  let html = "", inList = false, inCode = false, table: string[][] = [];
  const inline = (s: string) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/\*([^*]+)\*/g, "<i>$1</i>").replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !r.every((c) => /^:?-+:?$/.test(c.trim())));
    html += `<table class="tbl"><thead><tr>${rows[0].map((c) => `<th>${inline(c.trim())}</th>`).join("")}</tr></thead><tbody>${rows.slice(1).map((r) => `<tr>${r.map((c) => `<td>${inline(c.trim())}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    table = [];
  };
  for (const l of lines) {
    if (l.startsWith("```")) { html += inCode ? "</pre>" : "<pre>"; inCode = !inCode; continue; }
    if (inCode) { html += esc(l) + "\n"; continue; }
    if (/^\s*\|/.test(l)) { table.push(l.trim().replace(/^\||\|$/g, "").split("|")); continue; }
    flushTable();
    if (/^\s*[-*] /.test(l)) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${inline(l.replace(/^\s*[-*] /, ""))}</li>`; continue; }
    if (inList) { html += "</ul>"; inList = false; }
    const h = /^(#{1,4}) (.*)$/.exec(l);
    if (h) html += `<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`;
    else if (l.trim()) html += `<p>${inline(l)}</p>`;
  }
  flushTable();
  if (inList) html += "</ul>";
  return html;
}
