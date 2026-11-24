# Inference server (`src/server`)

`npm start` serves the lab and the API on **http://127.0.0.1:5177**. It binds to localhost only, because the network lives on this PC.
Options: `--port`, `--host`, `--model models/<run>`, `--web <dir>`, `--random`.

The model is loaded from `models/release`. If that is missing, the server uses the newest `models/<run>`. If there is no checkpoint at all, it uses a random init of the benchmarked preset and reports a warning in `/api/model` and in the WS `hello` message.
Static files are served from `web/dist`, then `web/`, then a placeholder page.

## HTTP

| Route | Returns |
|---|---|
| `GET /api/health` | `{ok, model}` |
| `GET /api/model` | `config`, `numParams`, `kernel`, `tokenizer` info, `pca.explained`, `warnings`, and `tensors[]`. Each tensor has `name`, `shape`, `offset`, `numel`, `mean`, `std`, `min`, `max`, `absMean`, `nearZero` and a 32-bin `hist` |
| `GET /api/weights/:name?format=json&maxRows=128&maxCols=128&mode=mean\|absmax` | A heatmap block-pooled to at most `maxRows x maxCols`: `{rows, cols, pool, min, max, data[]}` plus the tensor's stats |
| `GET /api/weights/:name?format=bin` | Raw little-endian Float32, with the shape in the `X-Shape` header |
| `POST /api/tokenize {code, cursor?}` | `{tokens[{id,text}], truncated, prefixChars}` |
| `POST /api/complete {code, cursor?, maxTokens=16, topK=10, temperature=0, stopAtNewline=true, seed?}` | `{completion, stopReason, prompt, steps[{id,text,p,entropy,topk[]}], ms}` |

The prompt is `<|bof|>` followed by the encoded text before `cursor`. If it is longer than the context, the server keeps the last `ctx` tokens and reports `truncated`.

## WebSocket `/api/trace`

The client sends `{type:"trace", reqId, code, cursor?, maxTokens?, topK?, attnWindow?, mlpTopK?, lastVectors?, stopAtNewline?, debounceMs?}`, or `{type:"cancel"}`.
Requests on one connection are debounced (60 ms by default) and the latest one wins: a newer request cancels an older one, even in the middle of a generation.

The server replies in this order:
1. `hello`: model label, config and warnings, sent once when the connection opens.
2. `tokens`: `{tokens[{id,text}], truncated, prefixChars}`.
3. `trace`: one traced forward pass over the window (`T` tokens, `S = 1 + 2L` stages: `embed`, `L0.attn`, `L0.mlp`, ...).
   - `pca[S][T*3]` and `norms[S][T]`: the residual stream in a fixed 3-D PCA basis.
   - `lastVectors[S][d]`: the full residual vectors of the last token.
   - `attention.last[L][H][T]`: the last query's attention over all keys. `attention.matrices[L][H][W*W]`: the full matrices for the last `W` tokens, starting at token `windowStart`.
   - `mlp.lastTop[L][{i,v}]`, `mlp.argmax[L][T]`, `mlp.argmaxV[L][T]`, `mlp.activeFrac[L]`.
   - `logitLens[S]{topk, entropy}`: the last token's residual decoded through `ln_f` and the tied head.
   - `next{topk, entropy}`.
4. `step`: one message per generated token, `{i, id, text, p, entropy, topk[], ms}`.
5. `done` with `{completion, stopReason, traceMs, totalMs}`, or `cancelled`.

## Measured on the Ryzen 7 5700U (`npm run smoke`, random-init `base` model, other jobs loading the CPU)

| Case | Trace frame | Trace round-trip | 16 tokens done |
|---|---|---|---|
| 19-token prefix | 32-34 KB | 48-69 ms | 93-140 ms (8 tokens) |
| Full 128-token window (prefix of more than 1,000 tokens) | 70-73 KB | 150-420 ms | 235-470 ms |

The decisions behind these choices are D48-D53 in `DECISIONS.md`.
