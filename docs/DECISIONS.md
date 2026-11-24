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
| D12 | Decoder-only pre-LN transformer, d=128, 4 layers, 4 heads, MLP 512, ctx 128 (≈1.07M params) *(provisional; final size from `bench-matmul`)* | Budget arithmetic (SPEC §2): about 10 GFLOPS × 2.5 h ≈ 9·10¹³ FLOP ≈ 12–15M tokens for 1M params, about 12 tokens/param (Chinchilla-ish). TinyStories shows this scale is fluent in narrow domains |
| D13 | GELU MLP, not SwiGLU | One scalar per neuron gives honest "neurons lighting up" visuals, and the MLP is 33 % cheaper |
| D14 | Learned absolute positions, not RoPE | Never extrapolated beyond ctx; the position embedding is itself a nice visual tensor |
| D15 | Tied input/output embeddings | Saves 262k params (25 %) at no quality cost at this size (1608.05859) |
| D16 | LayerNorm (not RMSNorm), no dropout | Standard GPT-2 form; with under 1 epoch over the data there is no need for dropout (Cramming) |
| D17 | fp32 everywhere; checkpoint = JSON header + raw float32 | Simple, and zero-copy into Float32Array in both Node and the browser. 1M params = 4.3 MB, small enough to commit |

## Engine
| # | Decision | Reason |
|---|---|---|
| D18 | Tape-based reverse-mode autograd over Float32Array, with fused kernels for attention, LayerNorm and cross-entropy | Fused ops avoid allocating T×T intermediates per op and are much faster in JS than generic elementwise graphs |
| D19 | Cache-blocked, register-tiled matmul; a WASM SIMD kernel only if > 1.5× faster *(measured)* | V8 doesn't vectorise JS loops. Hand-written WASM SIMD is still "from scratch", but we only add the complexity if it pays |
| D20 | Data-parallel `worker_threads` with SharedArrayBuffer weights/grads (W=8 default) | At about 1M params a full grad all-reduce is about 4 MB per step, which is cheap in shared memory. Data-parallel has far less sync than matmul-parallel at these small shapes. Matmul-parallel is the fallback |
| D21 | Finite-difference grad checks for every op | A hand-written engine is only credible if it is tested; bugs here would make silently wrong models |

## Training
| # | Decision | Reason |
|---|---|---|
| D22 | `rho = elapsed/budget` drives LR and curricula; warmup 200, then cosine in rho to 10 % | Copied from TinyMind `train.py` (`lr_at`): any budget ends annealed. Cramming does the same |
| D23 | Context curriculum 64→128 at rho 0.3; file curriculum (short and simple files first until rho 0.2) | The cheap-early-steps idea of CGAR (used by TinyMind) carried over to LMs; easy files are the "clean textbook" signal (phi-1) without an LLM |
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
