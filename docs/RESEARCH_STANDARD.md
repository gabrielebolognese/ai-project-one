# What made the TinyMind research work strong (and what we copy)

Source: the TinyMind research project (a separate repository, read-only here), studied 2026-11-24.
Files read: `README.md`, `results/survey.md`, `results/results.md`, `results/results.json`,
`results/stability_ablation.json`, `results/eval_log.txt`, `results/report.html` (section
structure), `tinymind/train.py`, `tinymind/bench_gemm.py`, `tinymind/build_report.py`,
`tinymind/server.py`, `web/index.html`, `web/app.js`, `web/style.css`.

## The practices

1. **A paper survey filtered by what this laptop can do.** `results/survey.md` queried the
   Hugging Face Papers API on 40 topics (1,894 papers), ranked them by upvotes and read the
   repos for platform support. Each candidate has a verdict column ("why it fits this
   laptop" or why not, e.g. "blocked only by disk: 42 GB > 40 GB cap"). Papers that were
   picked are mapped to the code that uses them.
2. **Hardware measured before the model was sized.** `bench_gemm.py` measured about 240
   GFLOPS fp32 GEMM on 8 threads, but only about 100 GFLOPS on the iGPU for small MLP-shaped
   matmuls. It benchmarked the *actual layer shapes*, not only square GEMMs.
   `bench_throughput.py` measured puzzles/s per config. The "hardware lesson" section turns
   this into a compute gap ("~1000x less than the paper") and a strategy.
3. **Budget-aware training.** In `train.py`, `rho = elapsed / budget` drives a warmup plus a
   cosine LR decay to a 10 % floor, the CGAR depth curriculum and a difficulty curriculum
   (by rating: easy until rho=0.1, full mix after 0.6). So "30 min" and "6 h" both end
   annealed. EMA weights (0.999), AdamW with no decay on norms/embeddings, grad clip 1.0,
   periodic eval every N minutes, `best.pt` kept on held-out score, and `log.jsonl` +
   `config.json` per run.
4. **Held-out evaluation with honest tables.** `evaluate.py` writes `results/results.json`
   and `results.md`: per-bucket accuracy re-weighted to the real test mix, plus **ms/puzzle
   latency**, a classic-solver baseline comparison, a sigma sweep and an oracle bound
   (pass@K) next to the practical selector.
5. **Ablations and negative results stated plainly.** `stability_ablation.json` explains why
   plain TRM did not learn (12 % → 41 % empty-cell accuracy with four named fixes). The README
   says what failed: "DIS's intermediate targets trained slower", "network-guided search did
   not reduce guesses on hard puzzles", plus a "What it does not do" section in the report.
6. **Everything local, one-command start.** `TinyMind.bat` → FastAPI server → web app with
   Think / Train / Research tabs. The front end has no external dependencies and works
   offline. The Train tab starts budgeted background runs, and the Research tab shows the
   result tables and live training curves.
7. **A self-contained report page.** `build_report.py` embeds results JSON, eval curves and a
   *real* model trace (a solved puzzle replay) into `report.html`, so it can be shared without
   a server.
8. **A README that reads like a short paper.** Quick start, the papers table ("what TinyMind
   takes from it"), "what is new here", layout, results table, findings, negative results.

## How the Java lab maps these

| TinyMind practice | Java Autocomplete Lab equivalent |
|---|---|
| HF Papers survey with hardware fit | `docs/survey.md`, same method and verdict column |
| `bench_gemm.py` at layer shapes | `scripts/bench.ts`: our own TS matmul at real shapes, 1/2/4/8 workers → sets model size |
| `rho`-driven LR + curricula | `src/train/schedule.ts`: rho drives warmup/cosine-to-10 % LR, context-length curriculum and file-length curriculum |
| EMA, best.pt, log.jsonl, config.json | Same: `models/<run>/{best.bin, last.bin, config.json, log.jsonl}` |
| Held-out buckets + ms/puzzle | Held-out *repositories*, top-1/top-5, line exact-match, ms/completion, n-gram baseline |
| Stability ablation + negative results | `results/ablations.json` + an explicit "Negative results" README section |
| Think / Train / Research tabs | Lab / Weights / Train / Research views |
| report.html with real trace | `results/report.html` with a real recorded completion trace |
