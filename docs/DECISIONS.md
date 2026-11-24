# Decisions

Each choice has a reason. Choices marked *(provisional)* are confirmed or changed by a measurement,
and the measurement is recorded here when it exists.

## Scope and standard
| # | Decision | Reason |
|---|---|---|
| D1 | The TinyMind research folder is the quality bar: survey, measured benchmark, budget-aware training, held-out tables, ablations, negative results, local app, report page | The owner asked to do it "the same way". See `RESEARCH_STANDARD.md` |
| D2 | Pure TypeScript end to end, with no ML framework (no tfjs, onnxruntime or Python) and nothing copied from TinyMind | "From absolute scratch" and "a ts project that makes a neural network on this pc". Writing the engine ourselves also means every tensor the lab shows is one we own |
| D3 | CPU only; the Vega iGPU is not used | The owner says "trainable on my mid pc". TinyMind found the iGPU slow at small MLP shapes (~100 GFLOPS), and WebGPU/DirectML from Node adds fragile native deps |
| D4 | MIT licence; data not committed; `data/SOURCES.md` + a re-download script | The repo goes on GitHub. Third-party code stays in its own repos with their licences recorded |

## Data
| # | Decision | Reason |
|---|---|---|
| D5 | Corpus = shallow clones of about 15–25 Apache-2.0/MIT Java repos | Auditable licences, no gated datasets, re-downloadable. The Stack is too big and gated |
| D6 | Split by **repository** (2–3 test repos, 1 val repo) | File-level splits leak project-specific identifiers. Unseen repos measure real generalisation, like TinyMind's held-out test puzzles |
| D7 | Strip licence headers; dedup exact and near-exact; cap each repo at 15 % of tokens | Boilerplate and one giant repo would dominate a 1M-param model |
| D8 | CRLF→LF, tabs→4 spaces | Fewer distinct whitespace tokens; consistent indentation |

## Tokenizer
| # | Decision | Reason |
|---|---|---|
| D9 | Byte-level BPE, vocab 2048 *(provisional; 1024 vs 2048 ablated)* | Open-vocabulary identifiers (2004.06343). 2048 × 128 embedding = 262k params is affordable, while 4096 would make the embedding half the model |
| D10 | Java-aware pre-tokenisation; merges don't cross identifier/operator/whitespace boundaries | TokDrift (2510.14972): grammar-misaligned subwords hurt code models; it also makes token chips readable in the lab |
| D11 | Runs of indentation spaces are single tokens | Indentation is very predictable; spending context positions on single spaces wastes the 128-token window |

## Model
| # | Decision | Reason |
|---|---|---|
| D12 | Decoder-only pre-LN transformer, d=128, 4 layers, 4 heads, MLP 512, ctx 128 (1,071,872 params, preset `base`) **(confirmed by measurement, see D38)** | `npm run bench`: with 8 workers `base` trains at 2,660 tok/s, about 23.2M tokens in 150 min, or 21.7 tokens/param (Chinchilla-optimal is about 20). `large` (2.2M) reaches only 5.3 tokens/param; `small` (0.54M) reaches 71 |
| D13 | GELU MLP, not SwiGLU | One scalar per neuron gives honest "neurons lighting up" visuals, and the MLP is 33 % cheaper |
| D14 | Learned absolute positions, not RoPE | Never extrapolated beyond ctx; the position embedding is itself a nice visual tensor |
| D15 | Tied input/output embeddings | Saves 262k params (25 %) at no quality cost at this size (1608.05859) |
| D16 | LayerNorm (not RMSNorm), no dropout | Standard GPT-2 form; with under 1 epoch over the data there is no need for dropout (Cramming) |
| D17 | fp32 everywhere; checkpoint = JSON header + raw float32 | Simple, and zero-copy into Float32Array in both Node and the browser. 1M params = 4.3 MB, small enough to commit |

## Engine
| # | Decision | Reason |
|---|---|---|
| D18 | Tape-based reverse-mode autograd over Float32Array, with fused kernels for attention, LayerNorm and cross-entropy | Fused ops avoid allocating T×T intermediates per op and are much faster in JS than generic elementwise graphs |
| D19 | Cache-blocked, register-tiled matmul **plus a WASM SIMD kernel, adopted: measured 6.1x faster** | Single thread at the real layer shapes: JS 4x4-tile kernel 1.3-1.6 GFLOPS, naive i-k-j JS 0.38, WASM SIMD (f32x4, 2x4 register tile, B-panel blocking) 7.2-9.6 GFLOPS. The module is assembled byte by byte in `src/engine/wasm-gemm.ts`: no compiler, no wabt, no framework. JS stays as a tested fallback (`JAL_KERNEL=js`) |
| D20 | Data-parallel `worker_threads` with SharedArrayBuffer weights and per-worker gradient slabs, **W=8 default** | Measured scaling for `base`: 1 worker 679 tok/s, 2 → 1,270, 4 → 2,226, 8 → 3,742, 16 (SMT) → 4,446. 16 is +19% but saturates the laptop, so 8 stays the default and `--workers 16` is opt-in. The all-reduce (mean of W slabs) plus AdamW and EMA run on the main thread while workers idle, so no locks are needed |
| D21 | Finite-difference grad checks for every op | A hand-written engine is only credible if it is tested; bugs here would make silently wrong models |

## Training
| # | Decision | Reason |
|---|---|---|
| D22 | `rho = elapsed/budget` drives LR and curricula; warmup 200, then cosine in rho to 10 % | Copied from TinyMind `train.py` (`lr_at`): any budget ends annealed. Cramming does the same |
| D23 | Context curriculum 64→128 at rho 0.3; file curriculum (short and simple files first until rho 0.2) | The cheap-early-steps idea of CGAR (used by TinyMind) carried over to LMs; easy files are the "clean textbook" signal (phi-1) without an LLM (the context curriculum is implemented in `src/train/schedule.ts`; the file curriculum needs per-file metadata from the corpus step and is not implemented yet) |
| D24 | AdamW (0.9, 0.95), wd 0.1 on matrices, clip 1.0, peak LR ≈ 3e-3 *(provisional; 5-min sweep incl. WSD)* | Standard small-LM recipe; MiniCPM shows high LR + WSD can beat cosine at small scale, so it is tested |
| D25 | EMA 0.999 for eval/release; `best.bin` on val loss; eval every 10 min | TinyMind's EMA + best-checkpoint practice gives a smoother, better final model |
| D26 | Release budget 150 min; ablations 20 min each, ≤ 1 h total | Within the owner's 2–3 h "mid PC" budget; ablations mirror TinyMind's stability ablation |

## Evaluation
| # | Decision | Reason |
|---|---|---|
| D27 | Metrics: loss/bpb, top-1/top-5, line exact match + edit similarity, identifier top-1, latency p50/p95 | CodeXGLUE/IntelliCode-style completion metrics; bpb compares across vocab sizes; latency mirrors TinyMind's ms/puzzle |
| D28 | Baseline: 4-gram interpolated Kneser-Ney over the same BPE tokens (+ bigram) | Hellendoorn & Devanbu: tuned n-grams are strong on code, so a fair, hard bar |
| D29 | Every result goes in `results/*.json` and the README, including where the model loses | TinyMind's honesty standard: negative results are reported, not hidden |

## Server and web lab
| # | Decision | Reason |
|---|---|---|
| D30 | Node `http` + `ws`, a binary WebSocket frame (JSON header + Float32 payload), latest-wins, KV cache for incremental typing | Per-keystroke traces need low latency; JSON number arrays would be about 5× larger and slower to parse |
| D31 | Stream embeddings, residuals, attention (all heads), MLP activations, logit lens and top-20 logits; tensors on demand by name + LOD | These are exactly what the views draw; big weight matrices load lazily |
| D32 | Vite + TS, CodeMirror 6 with lang-java, Canvas2D (+WebGL2 for big heatmaps), a hand-written tween engine, KaTeX labels, no UI framework | Full control over 3b1b-style animation; small bundle; works offline after build |
| D33 | 3b1b visual language: dark `#0e1116`, blue/yellow/teal/red palette, diverging colormap, 300–600 ms ease-in-out tweens, "write-in" strokes | The owner asked for "3 blue one brown animation style" and a "professional research lab" look |
| D34 | Views: Lab (editor + live network), Weights (every tensor, zoomable heatmaps, embedding PCA), Train, Research | Mirrors TinyMind's Think/Train/Research tabs and adds the requested weights browser |
| D35 | Port 5177, `npm start` + `Lab.bat` | One-command start, like `TinyMind.bat` |

## Layout
| # | Decision | Reason |
|---|---|---|
| D36 | `src/{engine,model,tokenizer,train,eval,server}`, `web/`, `scripts/`, `models/`, `results/`, `data/` (gitignored), `docs/` | Each step of the plan owns one folder; the engine is reusable by both trainer and server |
| D37 | Run TS with `tsx` in dev, `tsc` for type checking, ESM, strict | No build step needed for scripts; strict types catch shape bugs early |

## Measured (engine step, 2026-11-24)
| # | Decision | Reason |
|---|---|---|
| D38 | Release size rule: the **largest of three candidates that sees >= 10 tokens/param in 150 min**. Result: `base` | `results/bench.json`: small 4,445 tok/s (71 tok/param), base 2,660 tok/s (21.7), large 1,335 tok/s (5.3), all at 8 workers x 8 seqs x 128 tokens. The decision uses the slower candidate-pass number; the scaling pass measured base at 3,742 tok/s a minute later (15 W laptop power/thermal variance) |
| D39 | The benchmark records background CPU load, and the training step **re-runs `npm run bench` right before the release run**; `train.ts` reads `choice.preset` and `recommendedWorkers` from `results/bench.json` | The owner's laptop runs other jobs. A genuinely contended run (video renders + local transcription) measured base at 768 tok/s (3.5x lower, `results/bench-contended-run.json`), where the same rule would pick `small`. A 100% load reading alone is not proof of contention: spin-waiting pools read as busy but yield, and kernel GFLOPS (~9 for WASM) are the real health signal |
| D40 | Attention runs per (sequence, head) as GEMMs on the full T x T score square, with the causal mask applied in the softmax | Twice the arithmetic of a triangular loop, but on the ~6x faster SIMD kernel. Mask tests and grad checks pass on both the JS and WASM paths |
| D41 | Token shards are raw little-endian uint16 (`data/train.bin`, `data/val.bin`), with optional `data/meta.json {vocabSize}` | Zero-copy into a SharedArrayBuffer shared by every worker; vocab <= 65,535. The tokenizer step writes this format |
| D42 | Checkpoints: `models/<run>/weights.bin` (best EMA weights on val loss), `last.bin` (raw final weights), `config.json` (model + train options + a `tensors` table of name/shape/offset/numel), `log.jsonl`, `summary.json` | The web lab lists every tensor from the table and reads it zero-copy |
| D43 | Hardware lesson | Plain JS on V8 tops out at ~1.5 GFLOPS/thread (no auto-vectorisation). Hand-assembled WASM SIMD reaches ~9. End to end, the non-GEMM work (exp in softmax/cross-entropy, GELU, LayerNorm, gathers, GC) is about half of a single-thread step, and 8 workers sustain ~18-25 GFLOPS: the 15 W 5700U is power-limited well before 8x linear scaling |
| D44 | Corpus = 22 repos (18 train, 1 val = commons-codec, 3 test = jsoup, gson, retrofit); `junit-team/junit4` rejected (EPL), `caffeine` and `zxing` dropped (clones stalled) | Licences are read from each LICENSE file and the script refuses anything but Apache-2.0/MIT. Result: 24.2M train tokens (SPEC target 15-30M), 0.80M val, 1.68M test. Test repos cover parsing, serialisation and a REST client, none of which appear in training |
| D45 | Tokenizer: byte-level BPE, vocab 2048, ids 0-2 specials / 3-258 bytes / 259+ merges; pre-token regex keeps `newline+indent` as one piece and a single leading space glued to the next word | Measured bytes/token on the held-out test repos: 2.60 (vocab 1024), 3.08 (2048), 3.48 (4096). 2048 keeps the embedding at 262k params (D9) while 4096 would cost another 262k for +13% context reach. 100% of files in all three splits round-trip exactly |
| D46 | Per-repo cap of 15% is applied to training bytes (iterated to a fixed point, 12.21 MB), files chosen in hash order; val/test are not capped | Stops Guava, Jackson and RxJava dominating, without biasing the held-out sets |
| D47 | Near-duplicate key = hash of the first 2 KB of the body with package/import lines removed; train repos are processed first so val/test lose duplicates, never train | Keeps the held-out sets leak-free: measured 0 exact-duplicate files between train and val/test |

## Server step (2026-11-24)
| # | Decision | Reason |
|---|---|---|
| D48 | **Deviation from D30:** trace frames are plain JSON with numbers rounded to 3 decimals and condensed content, not a binary JSON-header + Float32 frame | Measured with `npm run smoke`: 32-34 KB per trace frame at 19 tokens, 70-73 KB at the full 128-token window, `JSON.stringify` about 3 ms. The forward pass (60-150 ms, depending on background load) dominates, so a binary codec would add complexity for no visible latency gain. Raw Float32 is still available per tensor via `GET /api/weights/:name?format=bin` |
| D49 | Condensed trace content: 3-D PCA points + L2 norms for every token at every residual stage (embed, then after each attn and each MLP); full d-dim vectors for the last token only; the attention row of the last query for every head over all keys, plus full matrices for the last `attnWindow` (16) tokens; top-12 MLP neurons of the last token + the argmax neuron per token; logit lens (top-5) per stage for the last token; top-k of the next token | Full tensors would be about 0.6 MB per keystroke at T = 128 (82k residual floats + 262k attention floats). This keeps what the views animate and fits in about 70 KB |
| D50 | One shared PCA basis for every layer, fit once at load on 4 x 128 tokens of `data/val.bin` (or a built-in Java snippet if the data is missing) | Fixed axes stop the points jumping between keystrokes, and a token's path through the layers can be compared within one space |
| D51 | Generation uses a KV-cached `IncrementalDecoder` built from the engine's own ops; when the cache is full it rebuilds on the last ctx - 32 tokens (context shift) | Recomputing the full window per token measured about 87 ms/token at T = 128, so 16 tokens took 1.39 s. The cache brings that to 235 ms in total. `server.test.ts` checks the cached logits against `Transformer.forward` (max diff < 1e-4) |
| D52 | Per-connection debounce (60 ms default, client can set `debounceMs`) and latest-wins cancellation checked between generated tokens; the server yields to the event loop after each token | A burst of keystrokes runs once (smoke: 4 keystrokes -> 1 run). A newer prefix stops a running 128-token generation within one token |
| D53 | Model resolution is `models/release`, then the newest `models/<run>` by `weights.bin` mtime, then random init of the benchmarked preset with a warning in `/api/model` and the WS hello | The server and the lab can be built and tested before the release run finishes |

## Web lab (step: 3Blue1Brown-style research-lab web app)

| # | Decision | Reason |
|---|---|---|
| D54 | Ghost text and the "Greedy completion" panel come from `POST /api/complete`; the WS trace is sent with `maxTokens: 0` so the socket only runs the traced forward pass. The server skips generation for `maxTokens: 0` and stops `/api/complete` generation as soon as the client drops the connection; the lab aborts the previous fetch on every keystroke | One generation per pause instead of two per keystroke on a single-threaded server, and both paths cancel stale work |
| D55 | The client renders a trace if its `reqId` is newer than the last rendered one (monotonic), not only if it equals the latest request sent | With strict "latest only", continuous typing rendered 2 of 226 traces (each finished pass was already superseded by the next keystroke). Monotonic acceptance rendered 88-104 of 226 in the same test and is never more than one pass behind; the server still debounces and supersedes (122-138 of 226 never computed) |
| D56 | Trace payload gains `mlp.lastAll` (all 512 GELU activations of the last token per layer) and `mlp.windowTop` (top-8 neurons of every window token) | The neuron grid lights up real activations; the token-by-token replay needs per-token neurons. About 14 KB more per frame |
| D57 | Embedding columns are computed in the browser from the raw `tok_emb` and `pos_emb` tensors (`E[id] + P[t]`), fetched once | Exact values with no per-keystroke payload |
| D58 | `GET /api/notebook` (run logs, results, bench, corpus, web perf, optional MODEL_CARD.md) and `GET /api/vocab`. Missing `results/results.json` is reported as null and the UI says "not measured yet" | The lab notebook shows only numbers the scripts wrote |
| D59 | Canvas2D with a hand-written tween engine; the scene renders straight to the screen while tweening and is cached in an offscreen canvas when idle (idle frames = one blit + flow particles) | Scene JS is about 6 ms per frame; the rest is rasterization. Measured in headless Chromium: about 60 fps animating and idle with GPU raster (ANGLE/D3D11 on the Vega iGPU), 28 fps animating and 50 idle with software SwiftShader. `npm run screenshots` uses GPU raster and records the renderer in `results/web-perf.json` |
| D60 | `npm run screenshots` starts the server as a separate process (like `npm start`) | In-process, the forward passes stalled Playwright's keyboard (226 characters took 87 s instead of about 20 s), which made the typing and fps measurements meaningless |
