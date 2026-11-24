import "./style.css";
import { Completer, TraceClient, getJson, getTensor, type CompleteResult, type ModelInfo } from "./api";
import { createEditor, type LabEditor } from "./editor";
import { NetworkView } from "./views/network";
import { WeightsView } from "./views/weights";
import { NotebookView, type Notebook } from "./views/notebook";
import { fmtInt } from "./theme";

const STARTER = `import java.util.ArrayList;
import java.util.List;

public class Inventory {
    private final List<String> items = new ArrayList<>();

    public void add(String item) {
        if (item == null) {
            throw new IllegalArgumentException("item must not be null");
        }
        items.add(item);
    }

    public int size() {
        return items.`;

type Tab = "lab" | "weights" | "notebook";

const $ = <T extends HTMLElement>(sel: string) => document.querySelector<T>(sel)!;

document.querySelector<HTMLDivElement>("#app")!.innerHTML = `
<header class="top">
  <div class="brand"><svg width="26" height="26" viewBox="0 0 26 26"><circle cx="5" cy="20" r="2.6" fill="#58C4DD"/><circle cx="13" cy="6" r="2.6" fill="#F7D96F"/><circle cx="21" cy="17" r="2.6" fill="#5CD0B3"/><path d="M5 20 Q 9 4 13 6 T 21 17" stroke="#58C4DD" stroke-width="1.4" fill="none" opacity=".8"/></svg>
    <div><b>Java Autocomplete Lab</b><span>a ${"1M"}-parameter transformer, trained and running on this PC</span></div></div>
  <nav class="tabs">
    <button data-tab="lab" class="on">Network</button><button data-tab="weights">Weights</button><button data-tab="notebook">Lab notebook</button>
  </nav>
  <div class="badges"><span class="badge" id="model-badge">loading model…</span><span class="dot" id="ws-dot" title="trace socket"></span></div>
</header>
<main>
  <section class="tab lab on" id="tab-lab">
    <div class="lab-left">
      <div class="panel editor-panel">
        <div class="panel-h"><span>Editor · Java</span><span class="hint"><kbd>Tab</kbd> accept · <kbd>Esc</kbd> dismiss</span></div>
        <div id="editor"></div>
      </div>
      <div class="panel caption-panel">
        <div class="panel-h"><span>What the network is doing</span><span class="hint" id="focus-label">following the last token</span></div>
        <div class="caption" id="caption">Type some Java on the left.</div>
      </div>
    </div>
    <div class="lab-right panel">
      <div class="net-toolbar">
        <div class="grp">
          <button class="btn" id="play" title="replay the forward pass stage by stage">▶ Replay pass</button>
          <button class="btn icon" id="step-back" title="previous stage">‹</button>
          <button class="btn icon" id="step-fwd" title="next stage">›</button>
          <button class="btn" id="play-tokens" title="move the query token by token">▶ Token by token</button>
        </div>
        <div class="grp">
          <div class="seg" id="mode"><button data-mode="tower" class="on">Tower</button><button data-mode="heads">Heads close-up</button></div>
        </div>
        <div class="grp">
          <label>layer <select id="layer"></select></label>
          <label>head <select id="head"></select></label>
          <label class="tog"><input type="checkbox" id="allpairs" checked> all query arcs</label>
        </div>
      </div>
      <div class="net-host" id="net"></div>
    </div>
  </section>
  <section class="tab weights" id="tab-weights"></section>
  <section class="tab notebook" id="tab-notebook"></section>
</main>
<footer class="status">
  <span id="st-model">–</span><span id="st-trace">trace –</span><span id="st-complete">complete –</span><span id="st-req">0 sent · 0 cancelled</span><span id="st-fps">– fps</span>
</footer>`;

let tab: Tab = "lab";
const net = new NetworkView($("#net"), {
  onHoverToken: (r) => (r ? editor?.highlight(r.from, r.to) : editor?.clearHighlight()),
  onCaption: (html) => { $("#caption").innerHTML = html; },
  onFocusChange: (label) => { $("#focus-label").textContent = label; },
});
const weights = new WeightsView($("#tab-weights"));
const notebook = new NotebookView($("#tab-notebook"));
const completer = new Completer();
let editor: LabEditor | null = null;
let lastComplete: CompleteResult | null = null;
let completeTimer: number | null = null;
let completeCount = 0;

const trace = new TraceClient({
  onStatus: (s) => { $("#ws-dot").className = `dot ${s}`; },
  onTrace: (m, tokens) => {
    net.setTokens(tokens);
    net.setTrace(m);
    $("#st-trace").textContent = `trace ${m.ms} ms · ${m.T} tokens`;
  },
});

function requestCompletion(code: string, cursor: number): void {
  if (completeTimer) clearTimeout(completeTimer);
  completeTimer = window.setTimeout(async () => {
    const r = await completer.complete(code, cursor);
    if (!r || !editor) return;
    if (editor.getCode() !== code || editor.getCursor() !== cursor) return; // stale
    lastComplete = r;
    completeCount++;
    net.setCompletion(r);
    $("#st-complete").textContent = `complete ${r.ms} ms · ${r.steps.length} tok`;
    if (r.completion.trim()) editor.showGhost(cursor, r.completion);
  }, 110);
}

function onEdit(code: string, cursor: number): void {
  completer.cancel();
  trace.trace(code, cursor);
  requestCompletion(code, cursor);
  $("#st-req").textContent = `${trace.stats.sent} sent · ${trace.stats.traces} rendered · ${trace.stats.cancelled} cancelled · ${trace.stats.dropped} stale dropped`;
}

function setTab(t: Tab): void {
  tab = t;
  document.querySelectorAll<HTMLButtonElement>(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
  document.querySelectorAll<HTMLElement>(".tab").forEach((s) => s.classList.toggle("on", s.id === `tab-${t}`));
  if (t === "notebook") notebook.show();
}
document.querySelectorAll<HTMLButtonElement>(".tabs button").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab as Tab)));

function setMode(m: "tower" | "heads"): void {
  net.setMode(m);
  document.querySelectorAll<HTMLButtonElement>("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
}
document.querySelectorAll<HTMLButtonElement>("#mode button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode as "tower" | "heads")));
$("#play").addEventListener("click", () => net.playPass());
$("#step-back").addEventListener("click", () => net.stepPass(-1));
$("#step-fwd").addEventListener("click", () => net.stepPass(1));
$("#play-tokens").addEventListener("click", () => net.playTokens());
$<HTMLInputElement>("#allpairs").addEventListener("change", (e) => { net.setAllPairs((e.target as HTMLInputElement).checked); });
$<HTMLSelectElement>("#layer").addEventListener("change", (e) => net.setLayer(Number((e.target as HTMLSelectElement).value)));
$<HTMLSelectElement>("#head").addEventListener("change", (e) => net.setHead(Number((e.target as HTMLSelectElement).value)));

// render loop + fps meter
let frames = 0, fpsT = performance.now(), fps = 0;
const fpsSamples: number[] = [];
function loop(now: number): void {
  if (tab === "lab") net.frame(now);
  else if (tab === "weights") weights.frame(now);
  else notebook.frame(now);
  frames++;
  if (now - fpsT >= 1000) {
    fps = (frames * 1000) / (now - fpsT);
    fpsSamples.push(fps);
    if (fpsSamples.length > 120) fpsSamples.shift();
    frames = 0;
    fpsT = now;
    $("#st-fps").textContent = `${fps.toFixed(0)} fps`;
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

async function boot(): Promise<void> {
  const info = await getJson<ModelInfo>("/api/model");
  const c = info.config;
  $("#model-badge").textContent = `${info.sourceLabel} · ${fmtInt(info.numParams)} params · ${c.nLayers}L d${c.dModel} ${c.nHeads}H`;
  if (info.source.kind === "random") $("#model-badge").classList.add("warn");
  $("#st-model").textContent = `${info.kernel} kernel · ctx ${c.ctx} · vocab ${c.vocabSize}`;
  $<HTMLSelectElement>("#layer").innerHTML = `<option value="-1">all</option>` + Array.from({ length: c.nLayers }, (_, l) => `<option value="${l}">${l + 1}</option>`).join("");
  $<HTMLSelectElement>("#head").innerHTML = `<option value="-1">all</option>` + Array.from({ length: c.nHeads }, (_, h) => `<option value="${h}">${h + 1}</option>`).join("");
  const [tok, pos, vocab] = await Promise.all([getTensor("tok_emb"), getTensor("pos_emb"), getJson<{ vocab: string[] }>("/api/vocab")]);
  net.setModel(c, tok.data, pos.data);
  weights.setModel(info, vocab.vocab);
  editor = createEditor($("#editor"), STARTER, (code, cursor) => onEdit(code, cursor), () => {});
  editor.focus();
  onEdit(editor.getCode(), editor.getCursor());
  getJson<Notebook>("/api/notebook").then((nb) => notebook.set(nb, info)).catch((e) => console.error(e));
}

// hooks for the screenshot / perf script (scripts/screenshots.ts)
(window as unknown as { __lab: unknown }).__lab = {
  setTab, setMode,
  net,
  weights,
  stats: () => ({ fps, fpsSamples: [...fpsSamples], trace: { ...trace.stats }, completions: completeCount, lastComplete: lastComplete?.completion ?? null, hasTrace: net.hasTrace }),
  focusEditor: () => editor?.focus(),
};

boot().catch((e) => {
  $("#caption").textContent = `Could not reach the inference server: ${(e as Error).message}. Start it with npm start.`;
});
