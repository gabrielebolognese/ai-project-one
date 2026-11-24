# Paper survey: a tiny Java code-completion transformer trained on a Ryzen 7 5700U CPU, in pure TypeScript

Date: 2026-11-24. Method: following the TinyMind survey (`research/results/survey.md`), we
queried the Hugging Face Papers API (`/api/papers/search?q=…` and `/api/papers/<arXiv id>`)
for topics including tiny transformer code completion, small-LM training efficiency, BPE for
code, transformer interpretability and visualisation, and small code LMs. Candidates were
filtered by one question: *can it be used, at our scale, on an 8-core laptop CPU with a
hand-written TS engine, in 2–3 hours?* Upvotes are HF Papers community upvotes as of today
(older papers predate the index, so their counts are low and not meaningful).

Hardware reality (to be measured in `results/bench.json`): TinyMind measured about 240 GFLOPS
fp32 with MKL on this CPU. Our hand-written JS kernels are expected at about 10–25 GFLOPS on 8
workers, so the compute budget is about 10¹⁴ FLOPs, roughly 10⁶ times less than a small
production code model.

## Picked: directly shapes the design

| Paper (arXiv) | Upvotes | Why it fits this laptop | How it is used |
|---|---|---|---|
| Attention Is All You Need (1706.03762) | 151 | The base architecture; at d=128, 4 layers it is about 1M params | Decoder-only transformer blocks, scaled dot-product attention |
| TinyStories: How Small Can LMs Be and Still Speak Coherent English (2305.07759) | 51 | Shows 1–10M-param transformers become fluent on a *narrow* domain. Java is a narrow, highly regular language | Justifies the scale; depth versus width at tiny size (prefer a few layers with small d) |
| Neural Machine Translation of Rare Words with Subword Units (BPE) (1508.07909) | 4 | BPE training is a simple counting algorithm and is cheap in TS | Byte-level BPE, vocab 2048, trained on the train split only |
| Code Completion using Neural Attention and Byte Pair Encoding (2004.06343) | 2 | BPE solves the open-vocabulary problem of identifiers for code completion | Confirms BPE over a closed identifier vocabulary |
| TokDrift: When LLM Speaks in Subwords but Code Speaks in Grammar (2510.14972) | 29 | Subword boundaries that cross grammar boundaries hurt code models | Java-aware pre-tokenisation; merges never cross operator/identifier boundaries |
| Using the Output Embedding to Improve Language Models (1608.05859) | 1 | At vocab 2048 × d 128 the embedding is about 25 % of params; tying halves it | Tied input/output embeddings |
| Training Compute-Optimal LLMs (Chinchilla) (2203.15556) | 13 | Tells us how to split a fixed FLOP budget between params and tokens | Size chosen by the benchmark to reach about 10–20 tokens/param in 2.5 h |
| Scaling Laws for Neural Language Models (2001.08361) | 13 | Loss is a smooth function of compute; small runs predict big ones | 20-min ablation runs are used to pick the release config |
| Cramming: Training a Language Model on a Single GPU in One Day (2212.14034) | 0 | The closest analogue: fixed wall-clock budget, LR schedule tied to the budget | Budget-tied LR decay (as in TinyMind's `rho`); pre-LN; no dropout; drop biases where harmless |
| MiniCPM: small LMs with scalable training strategies (2404.06395) | 24 | Warmup-Stable-Decay schedule, high LR works for small models | LR sweep includes a WSD variant against cosine-in-rho; the winner is recorded in DECISIONS |
| Decoupled Weight Decay Regularization (AdamW) (1711.05101) | 4 | Standard, cheap optimizer with 2 extra floats per param | AdamW in TS, weight decay on matrices only |
| Gaussian Error Linear Units (1606.08415) | 0 | One activation per neuron makes neurons easy to visualise | GELU MLP (not SwiGLU), so the "neurons lighting up" view is meaningful |
| Transformer Explainer: Interactive Learning of Text-Generative Models (2408.04619) | 174 | An in-browser, interactive visual explanation of GPT-2 is the closest UI prior art | Stage-by-stage forward-pass playback and the attention/softmax views; we add live typing, a weights browser and our own trained model |
| AttentionViz: A Global View of Transformer Attention (2305.03210) | 1 | Visual patterns of attention heads | Per-head colours and attention-arc design |
| In-context Learning and Induction Heads (2209.11895) | 2 | Induction heads (copying earlier identifiers) appear even in 2-layer models, and Java reuses identifiers constantly | Research view: an induction-head score per head on repeated-identifier probes; a likely "finding" |

## Considered, not chosen (and why)

| Paper (arXiv) | Upvotes | Verdict |
|---|---|---|
| RoFormer / RoPE (2104.09864) | 20 | Better length extrapolation, but rotated Q/K are harder to visualise and we never exceed the trained context. Learned absolute positions instead |
| GLU Variants Improve Transformer (SwiGLU) (2002.05202) | 5 | A small quality gain, but a gated product makes "neuron activation" ambiguous in the visualisation, and it costs 50 % more MLP matmuls |
| Root Mean Square Layer Normalization (1910.07467) | 2 | Slightly cheaper; LayerNorm kept for the standard GPT-2 form. A candidate ablation if time allows |
| Efficient Training of LMs to Fill in the Middle (2207.14255) | 1 | FIM is valuable for editor completion (code after the cursor), but it splits a tiny model's capacity. Future work |
| Improving FIM Code Completions via Context & Curriculum Based Learning (2412.16589) | 0 | Same: FIM-dependent |
| IntelliCode Compose (2005.08025) | 3 | Production-scale (GPT-2 medium, multilingual); used only for its line-completion metrics (exact match, edit similarity) |
| CodeXGLUE (2102.04664) | 2 | Its code-completion tasks (token- and line-level, Java) define the metrics; the benchmark itself is too big to evaluate at our model's scale. We mirror the metric definitions on our held-out repos |
| The Stack (2211.15533) | 9 | A permissive-licence code dataset, but gated download and far more data than we can use. Shallow clones of named Apache/MIT repos are simpler and auditable |
| SantaCoder (2301.03988) | 7 | Data hygiene (near-dedup, licence filtering) is copied; the 1.1B model is not relevant |
| Textbooks Are All You Need (phi-1) (2306.11644) | 158 | Data quality beats quantity, but synthetic textbook data needs an LLM to generate, which is out of scope (no external models). We approximate it with a "clean code first" file curriculum |
| TinyLlama (2401.02385) | 93 | 1.1B params; far beyond CPU training |
| HARE: human priors for small-LM efficiency (2406.11410) | 39 | Data-selection ideas for 1B-scale models; not actionable at 1M params |
| LongCoder (2306.14893), ReACC retrieval completion (2203.07722) | 0, 0 | Long context and retrieval are strong ideas but add a second system. Retrieval (reusing identifiers from the current file) is a follow-up project |
| Transformer Interpretability Beyond Attention Visualization (2012.09838) | 1 | Relevance propagation needs an extra backward pass per trace; possible later as a "why this token" overlay |
| TensorLens (2601.17958) | 3 | High-order attention tensors are interesting but heavy for live 30 fps rendering |

## Not on arXiv but used
- Kneser & Ney (1995), *Improved backing-off for M-gram language modeling*; Chen & Goodman (1998),
  *An Empirical Study of Smoothing Techniques for Language Modeling*. These are the **n-gram baseline**.
- Hindle et al. (ICSE 2012), *On the Naturalness of Software*; Hellendoorn & Devanbu (FSE 2017),
  *Are deep neural networks the best choice for modeling source code?* These show that well-tuned
  n-grams are strong code baselines, so beating one is a meaningful bar.
- Elhage et al. (2021), *A Mathematical Framework for Transformer Circuits*, and nostalgebraist (2020),
  *interpreting GPT: the logit lens*. These underpin the residual-stream and logit-lens views.
- 3Blue1Brown, *Neural networks* series (ch. 5–7 on transformers). This is the visual language reference.
