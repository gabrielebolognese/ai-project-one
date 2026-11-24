# Java Autocomplete Lab: specification

Status: v1 spec, 2026-11-24. Written before any implementation code.
Companion docs: `docs/RESEARCH_STANDARD.md` (what we copy from the TinyMind research work),
`docs/survey.md` (papers), `docs/DECISIONS.md` (every choice with its reason).

## 1. Goal

A tiny neural network for **Java code completion** that is trained from scratch on the
owner's laptop CPU, then served to a **TypeScript web lab**. The lab shows every weight and
animates the network in a 3Blue1Brown style while you type. Everything is TypeScript:
tokenizer, tensor/autograd engine, model, trainer, inference server and web app. There is
no Python and no ML framework in this repo, and no code is copied from TinyMind.

The standard to match is the TinyMind research folder (`Desktop/research`). From it we copy
these concrete practices (details in `docs/RESEARCH_STANDARD.md`):

- a **paper survey with a hardware-fit verdict column** (HF Papers API / arXiv IDs);
- **throughput measured at the real layer shapes** before choosing the model size
  (TinyMind's `bench_gemm.py`, which found 240 GFLOPS square GEMM but much lower at MLP shapes);
- **budget-aware training**: progress `rho = elapsed / budget` drives the LR schedule and the
  curricula, so any wall-clock budget gives an annealed model; plus EMA weights, `best` checkpoint on
  held-out score, `log.jsonl` + `config.json` per run;
- **held-out evaluation tables** with latency and a **classic baseline** (TinyMind compared
  against a classic solver; we compare against a token n-gram model);
- **ablations and negative results** written down plainly, the way the TinyMind README says
  "DIS targets trained slower here" and "guided search did not reduce guesses on hard puzzles";
- a **one-command local app** that works offline, plus a **self-contained report page** with a real
  recorded trace.

Success criteria (all measured, all reported even if bad):
1. The release model beats the n-gram baseline on held-out-repo next-token top-1 accuracy.
2. Completion latency is below 50 ms for a 1-line greedy completion (about 8 tokens) on this CPU.
3. The lab animates a forward pass for every keystroke without dropping below 30 fps.
4. Training to release takes 2–3 CPU hours and is reproducible from scripts.

## 2. Hardware budget

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 5700U, 8 cores / 16 threads, Zen 2, AVX2 (no AVX-512) |
| RAM | 16 GB (keep training RSS < 3 GB; the owner keeps using the laptop) |
| GPU | Vega iGPU: **not used** (CPU only, by design) |
| OS | Windows 11, Node.js 22 (v22.16 installed) |
| Training budget | **2–3 hours wall-clock** for the release run, plus ≤ 1 h total of short ablations |
| Disk | Corpus raw ≤ 1.5 GB (gitignored); repo artifacts ≤ 50 MB |

**Realism check.** TinyMind measured about 240 GFLOPS with MKL. Hand-written JS (no SIMD
intrinsics in plain TS; V8 does not auto-vectorise Float32Array loops) is expected to reach
about **1.5–4 GFLOPS per thread** with a cache-blocked, transposed-B inner loop, or about
**10–25 GFLOPS on 8 workers**. Training cost ≈ 6·N·T FLOPs (plus about 20 % attention at ctx 128).
At a sustained 10 GFLOPS for 2.5 h (9·10¹³ FLOP), a 1.0M-param model sees about **12–15M
tokens**, roughly 12 tokens/param. That is close enough to compute-optimal (Chinchilla, 2203.15556)
for a tiny model. The exact size is chosen from `scripts/bench-matmul.ts` (§5.4), not guessed.

## 3. Package layout

```
java-autocomplete-lab/
  package.json, tsconfig.json   ESM, TypeScript strict, Node 22; tsx for running TS directly
  src/engine/     Tensor (Float32Array), autograd tape, ops, matmul kernels, worker pool, AdamW
  src/model/      config, transformer (embed, attention, MLP, LayerNorm), forward-with-trace, (de)serialise
  src/tokenizer/  Java pre-tokenizer + byte-level BPE trainer/encoder/decoder
  src/train/      dataset/packing, schedule (rho), trainer, EMA, checkpointing, eval loop
  src/eval/       held-out metrics, n-gram baseline, latency bench -> results/
  src/server/     Node HTTP + WebSocket inference server, activation tracing protocol
  web/            Vite + TypeScript lab (CodeMirror 6, canvas/WebGL renderer, tween engine)
  scripts/        fetch-corpus.ts, build-corpus.ts, train-tokenizer.ts, bench-matmul.ts,
                  train.ts, evaluate.ts, build-report.ts
  models/         release/ (committed if < 20 MB): model.bin, config.json, tokenizer.json, log.jsonl
  results/        results.json, results.md, ablations.json, bench.json, report.html
  data/           (gitignored) raw clones, cleaned corpus shards, token shards; SOURCES.md is committed
  docs/           SPEC.md, survey.md, DECISIONS.md, RESEARCH_STANDARD.md
```

## 4. Data and tokenizer

### 4.1 Corpus
- Shallow clones (`git clone --depth 1`) of **permissively licensed (Apache-2.0 / MIT) Java
  repos**, about 15–25 repos, for example Guava, Apache Commons (Lang, Collections, IO, Text),
  JUnit 5, Jackson core/databind, OkHttp (Java history), Spring PetClinic, Gson, Joda-Time,
  RxJava, Netty (subset), jsoup, Mockito, LeetCode-style solution repos with MIT licence.
  The final list and licences go in `data/SOURCES.md`, written by `scripts/fetch-corpus.ts`.
- Cleaning: only `*.java`. Drop files > 100 KB or < 200 B, generated code (`// Generated`,
  `@Generated`, protobuf), exact duplicates (hash) and near-duplicates (first 2 KB hash).
  Strip licence header comments, because they are repeated boilerplate that would dominate a tiny model.
  Normalise CRLF→LF and tabs→4 spaces.
- **Split by repository, not by file.** 2–3 whole repos are held out as `test` and 1 as `val`.
  This measures generalisation to unseen code bases, like TinyMind's held-out test puzzles.
  Target: about 40–80 MB of cleaned Java (≈ 15–30M BPE tokens) for training.

### 4.2 Tokenizer
- **Byte-level BPE** (Sennrich et al., 1508.07909) trained in TS on the train split. **Vocab 2048**
  (range explored: 1024 / 2048 / 4096, chosen by bytes-per-token versus embedding cost).
- Java-aware pre-tokenisation (regex): identifiers, camelCase boundaries kept inside merges,
  numbers, operators (`==`, `->`, `::`, `+=` …), and runs of spaces as their own tokens
  (indentation is very predictable and cheap to learn). Merges never cross pre-token
  boundaries, which partly avoids the subword/grammar misalignment described in TokDrift (2510.14972).
- Special tokens: `<|bof|>` (file start), `<|eof|>`, `<|pad|>`.
- Output: `models/<run>/tokenizer.json` (merges + vocab), and the bytes/token measured on val.

## 5. Model, engine and training

### 5.1 Architecture (default; final size set by the benchmark)

| Hyper-parameter | Default | Allowed range |
|---|---|---|
| Type | decoder-only, pre-LN transformer (GPT-2 style) | – |
| Vocab | 2048 | 1024–4096 |
| Context | 128 tokens (train), 128 inference window | 128–256 |
| d_model | 128 | 96–192 |
| Layers | 4 | 2–6 |
| Heads | 4 (head dim 32) | 2–6 |
| MLP | 4× (512), GELU | – |
| Positions | learned absolute (visualisable as a heatmap) | – |
| Norm | LayerNorm | – |
| Output | tied with token embedding (1608.05859) | – |
| Params | ≈ 1.07M (embed 262k + pos 16k + 4 × 197k) | 0.5–3M |

### 5.2 Tensor / autograd engine (`src/engine`)
- `Tensor { data: Float32Array, shape: number[], grad?: Float32Array }`, row-major and contiguous.
  The tape-based reverse-mode autograd records closures. Ops: matmul, add (broadcast bias), mul,
  GELU, LayerNorm (fused fwd/bwd), softmax with causal mask (fused attention per head),
  embedding gather/scatter-add, cross-entropy (fused log-softmax + NLL), reshape/transpose views
  where they are cheap.
- **Matmul kernel**: i-k-j loop order with B pre-transposed where useful, 4×4 or 4×8 register
  tiles, 64-column cache blocking, and `Math.fround`-free Float32Array accumulation. We also
  test a WASM SIMD variant (`v128`, hand-written `.wat`, no framework), adopt it only if it is
  more than 1.5× faster, and record the result in `results/bench.json`.
- **Parallelism**: `worker_threads` with `SharedArrayBuffer` weights. The primary strategy is
  **data-parallel**: each of W workers (W=8 default; 16 SMT threads are measured too) runs
  forward/backward on its micro-batch, writes gradients to its own shared slab, and the main
  thread sums them and runs AdamW (or the sum is tree-reduced in workers). Matmul-parallel
  (splitting rows of large matmuls) is the fallback if the benchmark shows data-parallel is
  memory-bound.
- Gradient check (finite differences) for every op is in `src/engine/*.test.ts`. CI-free, run by `npm test`.

### 5.3 Training (`src/train`), budget-aware like TinyMind
- **Progress** `rho = elapsed / budget` (budget in minutes is a CLI flag: `--minutes 150`).
- **LR**: linear warmup for 200 steps, then cosine in rho down to a 10 % floor (the same shape as
  TinyMind `lr_at`). AdamW β=(0.9, 0.95), wd 0.1 on matrices only, grad-clip 1.0, peak LR
  about 3e-3 (tiny models tolerate high LR; checked with a 5-min LR sweep).
- **Curriculum (driven by rho)**: (a) context length 64 → 128 at rho 0.3 (cheaper early steps,
  as in CGAR's shallow-first idea); (b) file curriculum: short, simple files first (under 4 KB,
  low nesting) until rho 0.2, then the full mix.
- Batch: 8 workers × 8 sequences × 128 tokens ≈ 8k tokens/step.
- **EMA** of weights (0.999) used for eval and release. Eval every 10 min on 200k val tokens.
  Keep `best.bin` on val loss. Log `log.jsonl` (`t, rho, step, tokens, lr, loss, val_loss,
  tok/s`) and `config.json`.
- The binary checkpoint format is a JSON header (names, shapes, offsets) followed by raw
  little-endian float32. The server and the browser read it with zero copying.
- **Ablations** (short 20-min budgets each, ≤ 1 h total): no curriculum; vocab 1024 vs 2048;
  2 vs 4 layers at equal time. Results go in `results/ablations.json`, even negative ones.

### 5.4 Benchmark first (`scripts/bench-matmul.ts`)
Measure GFLOPS of our kernel at the real shapes `(B·T × d) @ (d × 4d)`, `(B·T × d) @ (d × V)`
and attention `(T × 32) @ (32 × T)`, with 1/2/4/8/16 workers, plus end-to-end tokens/s for 3
candidate configs (0.5M / 1M / 2M). Pick the largest config that sees ≥ 10 tokens/param
within the budget. Write `results/bench.json` and a "hardware lesson" paragraph for the README.

## 6. Evaluation (`src/eval`, `scripts/evaluate.ts`)

All on the **held-out test repos**, never seen by the tokenizer or the model:

| Metric | Definition |
|---|---|
| Val/test loss, perplexity, bits-per-byte | bpb makes results comparable across vocab sizes |
| Next-token top-1 / top-5 accuracy | over all test tokens (window 128, stride 64) |
| Line-completion exact match | 1,000 sampled lines: given the file prefix and the first ⅓ of the line, greedy-complete to `\n`; exact match and edit similarity (normalised Levenshtein) |
| Identifier accuracy | top-1 restricted to positions where the target begins an identifier |
| Latency | ms per next-token forward and ms per full 1-line completion (p50 / p95), on 1 thread and in the server |
| **Baseline** | token n-gram (n=4 over the same BPE tokens, interpolated Kneser-Ney, with simple stupid-backoff as a sanity check), trained on the same train split |
| Extra baseline | "most frequent next token given previous token" (bigram) |

Outputs: `results/results.json`, `results/results.md` (tables like TinyMind's), and
`results/report.html` (self-contained: curves, tables and a real recorded completion trace
replayed with the lab's renderer). The README "Findings" and "Negative results" sections are
written from these numbers only.

## 7. Inference server (`src/server`)

- Node 22, plain `node:http` + the `ws` package (the only runtime dependency besides
  CodeMirror on the web side). Serves the built `web/dist` statically. `npm start` runs it at
  `http://127.0.0.1:5177` and opens the browser. Windows `Lab.bat` launcher.
- REST: `GET /api/model` (config, param count, tensor list with shapes and stats);
  `GET /api/tensor/:name?lod=` (raw float32 tensor bytes, or a downsampled LOD for big ones);
  `GET /api/results`; `GET /api/training/:run` (log.jsonl curves); `POST /api/train` / `stop`
  (background budgeted run, like TinyMind's Train tab).
- **WebSocket `/ws`**: the client sends `{type:"trace", id, text, cursor}` (debounced 60 ms).
  The server keeps a KV cache for the previous text prefix and recomputes only the changed tail.
  It replies with **one binary frame** (JSON header + Float32Array payload) containing:
  - `tokens`: ids + strings + byte spans of the last ≤ 64 context tokens;
  - `embed`: token-embedding and position-embedding vectors per token (T × d);
  - per layer ℓ: `resid_pre`, `resid_mid` norms and the full residual vectors (T × d);
    `attn[ℓ]`: attention probabilities for all heads (H × T × T, causal);
    `mlp[ℓ]`: post-GELU neuron activations (T × 512) and the top-16 neurons by |act| at the last position;
  - `logit_lens[ℓ]`: top-5 tokens from decoding each layer's residual through the final LN + unembedding;
  - `logits`: top-20 next tokens with probabilities (softmax at temperature 1);
  - `completion`: greedy ghost text (up to the end of the line, ≤ 24 tokens) streamed as
    `{type:"token"}` messages, so the UI can animate generation token by token.
  Payload ≈ 4 layers × (4×64×64 + 64×512 + 2×64×128) floats ≈ 0.7 MB per trace, fine on localhost.
- Budget: a trace plus completion in under 50 ms. The server uses the engine's forward without
  the tape (inference mode, preallocated buffers).

## 8. Web lab (`web/`), research-lab look in a 3Blue1Brown style

**Stack:** Vite + TypeScript, CodeMirror 6 (`@codemirror/lang-java`), Canvas2D for the main
scene (WebGL2 for heatmaps larger than 256×256), and a small hand-written tween/timeline engine
(easing `easeInOutCubic`, springs for hover). No UI framework.

**Visual language:** near-black background `#0e1116`, a 3b1b-like palette (blue `#58C4DD`,
yellow `#FFFF00`/`#F7D96F`, teal `#5CD0B3`, red `#FC6255`), a diverging blue↔black↔red colormap
for weights and activations, LaTeX-like labels (KaTeX for formulas such as
`softmax(QKᵀ/√d)V`), and smooth 300–600 ms tweens between states. Nothing snaps: when new
activations arrive, values interpolate from the old ones. Elements "write themselves in"
(stroke-dash animation) when a view first appears.

**Views (tabs):**
1. **Lab** (main). Left: the editor with **ghost-text completion** (Tab accepts, Esc dismisses,
   alternatives come from the top-5). Right: the network scene for the current cursor, laid out
   as columns from left to right:
   - **Tokens**: the last 16–32 context tokens as chips. Hovering a chip highlights its bytes in the editor.
   - **Embedding**: each token's 128-d vector as a vertical colour strip that slides in. Token and
     position embeddings are added with an animated "+" merge.
   - **Blocks 1…L**: each block shows **attention arcs** between token chips (arc thickness and
     opacity = attention probability, per-head colour, head selector, arcs animate as they grow
     out of the query token). Then the **MLP**: 512 neurons as a dot grid that **lights up**
     (glow radius ∝ activation), with the top neurons labelled. The **residual stream** is a
     horizontal "river" whose width = residual norm per layer.
   - **Logit lens**: under each block, the top-1 guess at that depth fades in, so you can watch
     the prediction form layer by layer.
   - **Output**: a **softmax bar chart** of the top-10 next tokens. Bars tween to new heights,
     and the chosen token flies into the editor as ghost text.
   - A playback scrubber replays the last forward pass stage by stage (embed → L1 attn → L1 MLP
     → … → softmax) with a step-through "explain" caption under each stage (3b1b narration style).
2. **Weights**: a browser for **every tensor** (tree list with shape, param count, mean/std,
   histogram). The selected matrix renders as a **zoomable heatmap** (wheel zoom, drag pan,
   LOD tiles from the server; hover shows `W[i,j]` exactly). Special renderings: the token
   embedding as a 2-D PCA/UMAP-like scatter of the vocab (computed in-browser with power-iteration
   PCA) with Java keywords highlighted; Q/K/V/O split per head; position embeddings as a
   cosine-similarity matrix.
3. **Train**: start a budgeted run (minutes, size preset), watch live loss/val curves, tokens/s
   and LR/rho. Same idea as TinyMind's Train tab.
4. **Research**: results tables from `results/results.json`, the ablations, the baseline
   comparison, latency, and links to the survey and decisions.

**Performance budget:** 60 fps target and 30 fps minimum. Interpolation runs in rAF. Heatmaps
are drawn to an offscreen canvas once per tensor. Trace frames are dropped if a newer one
arrives (latest-wins).

## 9. Deliverables and acceptance

- `npm run fetch-corpus && npm run build-corpus && npm run tokenizer && npm run bench &&
  npm run train -- --minutes 150 && npm run evaluate && npm run report && npm start`
  reproduces everything.
- README in the TinyMind format: quick start; the papers table ("what this project takes from
  it"); what is new; layout; results table vs baseline with latency; findings; **negative
  results**; hardware lesson; licence (MIT) and data sources.
- `models/release` committed if it is under 20 MB (≈ 4.3 MB at 1M params, fp32).

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| JS matmul below 5 GFLOPS on 8 workers | Shrink to about 0.5M params / 3 layers, use ctx 64→128, and try the WASM SIMD kernel |
| Corpus too repetitive (Guava dominates) | Cap tokens per repo at 15 % of the corpus |
| Model doesn't beat KN n-gram on top-1 | Report it honestly; also report top-5 and long-range (identifier reuse) where transformers usually win |
| Trace payload too slow | Send only the last 32 tokens and quantise to float16 (Uint16Array) |
