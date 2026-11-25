# Java Autocomplete Lab: evaluation

Held-out test repos (jhy/jsoup, google/gson, square/retrofit): 709 files, 1,677,739 tokens, never seen by the tokenizer or the trainer. Next-token metrics use 2,700 windows of 128 tokens spread evenly over the test stream, scoring the 96 positions per window that have at least 32 tokens of context (259,200 target tokens, identical for every method). Line completion uses 500 seeded random lines cut at a random token boundary (greedy decoding, stop at newline). Model: 1.07M parameters, trained 138.8 min on this laptop's CPU (Ryzen 7 5700U, 16 worker threads), 14.7M tokens seen (13.7 tokens/param).

## Main results

| Method | Params / size | Top-1 | Top-5 | Perplexity | Line exact match | Line edit sim | ms/completion (mean / p95) |
|---|---|---|---|---|---|---|---|
| transformer (release) | 1.07M (4.3 MB) | 49.5% | 70.1% | 12.89 | 19.4% | 0.427 | 81.0 / 167.8 |
| transformer (lab-preview) | 1.07M (4.3 MB) | 29.8% | 51.4% | 40.87 | 12.6% | 0.302 | 84.1 / 167.9 |
| 5-gram Kneser-Ney | 168 MB | 35.5% | 57.7% | 32.83 | 11.4% | 0.300 | 1.2 / 3.1 |
| 5-gram Kneser-Ney + file cache | 168 MB | 58.2% | 77.7% | 9.10 | 19.6% | 0.403 | 1.5 / 4.0 |
| 5-gram Kneser-Ney + window cache | 168 MB | 45.9% | 66.7% | 18.96 | 14.0% | 0.334 | 1.6 / 3.9 |
| most-frequent token | 0 MB | 4.5% | 10.7% | 662.58 | 0.0% | 0.013 | 0.0 / 0.0 |

Paired line comparison (transformer (release) vs 5-gram Kneser-Ney, same 500 lines): both exact 45, only network 52, only n-gram 12, neither 391; exact two-sided sign test on discordant lines p = 4.6e-7.

## Top-1 by test repo

| Method | jhy/jsoup | google/gson | square/retrofit |
|---|---|---|---|
| transformer (release) | 45.4% (ppl 17.91) | 52.5% (ppl 9.70) | 53.8% (ppl 10.11) |
| transformer (lab-preview) | 26.5% (ppl 56.52) | 33.0% (ppl 30.75) | 31.9% (ppl 32.49) |
| 5-gram Kneser-Ney | 31.5% (ppl 46.01) | 36.9% (ppl 28.00) | 42.8% (ppl 19.72) |
| 5-gram Kneser-Ney + file cache | 55.7% (ppl 11.55) | 58.9% (ppl 8.09) | 63.1% (ppl 6.41) |
| 5-gram Kneser-Ney + window cache | 42.9% (ppl 25.23) | 47.4% (ppl 16.02) | 50.6% (ppl 13.18) |
| most-frequent token | 4.2% (ppl 661.94) | 4.6% (ppl 669.19) | 5.1% (ppl 651.20) |

## Top-1 by target token class

| Method | identifier/keyword (63% of tokens) | punctuation/operator (23% of tokens) | whitespace (8% of tokens) | literal/other (5% of tokens) |
|---|---|---|---|---|
| transformer (release) | 42.6% | 63.5% | 68.0% | 41.3% |
| transformer (lab-preview) | 19.0% | 50.9% | 58.3% | 22.5% |
| 5-gram Kneser-Ney | 31.5% | 47.9% | 37.3% | 26.9% |
| 5-gram Kneser-Ney + file cache | 57.5% | 62.2% | 56.8% | 51.7% |
| 5-gram Kneser-Ney + window cache | 43.9% | 52.9% | 46.3% | 39.3% |
| most-frequent token | 0.0% | 19.4% | 0.0% | 0.0% |

## Line completion by line kind

| Method | code (n=435) EM / edit sim | comment (n=65) EM / edit sim |
|---|---|---|
| transformer (release) | 20.7% / 0.441 | 10.8% / 0.331 |
| transformer (lab-preview) | 13.3% / 0.309 | 7.7% / 0.253 |
| 5-gram Kneser-Ney | 11.3% / 0.297 | 12.3% / 0.326 |
| 5-gram Kneser-Ney + file cache | 20.5% / 0.413 | 13.8% / 0.336 |
| 5-gram Kneser-Ney + window cache | 14.0% / 0.337 | 13.8% / 0.311 |
| most-frequent token | 0.0% / 0.013 | 0.0% / 0.012 |

## Training runs

| Run | Params | Budget | Wall clock | Tokens seen | Tokens/param | Train tok/s | Peak LR | Best val loss (commons-codec) |
|---|---|---|---|---|---|---|---|---|
| transformer (release) | 1.07M | 150 min | 138.8 min | 14.66M | 13.7 | 1761 | 0.003 | 2.5925 (step 9875) |
| transformer (lab-preview) | 1.07M | 22 min | 22.8 min | 2.60M | 2.4 | 2213 | 0.003 | 3.6436 (step 572) |

Evaluated weights (sha256 of weights.bin):

- transformer (release): `d91725ce74defe8167bca25e692b043797b69717fe0a4a4a9cca8169406c77b2`; background load while timing line completion: machine 98.1% busy, 14.62 cores used by other processes
- transformer (lab-preview): `ab0cd68664f5b6e510d8f4547023af95088a5db3f91cc7b6f3fb53469a833415`; background load while timing line completion: machine 96% busy, 14.28 cores used by other processes

Note on transformer (release): The 150-minute run did not reach its end-of-run record. Its last surviving log line is at 138.8 min (step 10195, rho 0.925, LR 3.37e-4). The process was gone when this step was re-done, and the worktree's copy of its output folder had been removed by a branch reset between worker attempts. So the cause and the exact moment of termination were not observed. This log is a copy taken while the run was still going, so the record after 138.8 min is missing. weights.bin is the trainer's own best-on-validation EMA checkpoint, written by the trainer at step 9875 (130.3 min, val loss 2.5925 on commons-codec). The next scheduled evaluation was at about 140 min. last.bin and the optimizer state were never written. elapsedSec, steps and tokens are taken from the last surviving log line, and tokensPerSec includes evaluation time. Other processes on the laptop (including other Foreman tasks) were also using the CPU during the run, so throughput is below the bench figure.

## LR / batch pilots (6 min each, before the release run)

6-minute pilots, base preset (1.07M params), 16 worker threads, budget-aware warmup+cosine, ctx curriculum 64->128 at rho 0.3. Val = commons-codec (65,536 tokens). Run under background load from other processes.

| Run | Peak LR | Batch/worker | Tokens/step (T=128) | Steps | Tokens | tok/s | Best val loss (EMA) |
|---|---|---|---|---|---|---|---|
| pilot-lr3e-3 | 0.003 | 4 | 8192 | 139 | 0.89M | 2567 | 5.153 |
| pilot-lr6e-3 | 0.006 | 4 | 8192 | 132 | 0.86M | 2489 | 5.840 |
| pilot-lr3e-3-b2 | 0.003 | 2 | 4096 | 282 | 0.87M | 2455 | 4.416 |
| pilot-lr3e-3-b1 | 0.003 | 1 | 2048 | 569 | 0.90M | 2567 | 4.254 |

## n-gram baseline

Order 5, interpolated modified Kneser-Ney, built in 62.9 s from 24,197,288 train tokens; unique n-grams per order: 1,970 / 340,354 / 2,068,931 / 4,428,384 / 6,555,665. 168 MB in memory, against 4.3 MB for the network. The cache variant (Tu et al. 2014; Hellendoorn & Devanbu 2017) mixes in the tokens that followed the longest matching context (up to 3 tokens) earlier in the same file, up to 2048 tokens back, so it sees more of the file than the network's 128-token window. The window-cache variant only looks at the tokens the network also sees (the same window), which makes it the like-for-like comparison. Mixing weight total/(total+beta), beta=2 (file) and 2 (window), chosen on the validation repo from {0.5, 1, 2, 4, 8, 16}.

## Examples (first 12 sampled lines)

| Prefix (line) | Target | transformer (release) | transformer (lab-preview) | 5-gram Kneser-Ney | 5-gram Kneser-Ney + file cache | 5-gram Kneser-Ney + window cache | most-frequent token |
|---|---|---|---|---|---|---|---|
| `import io.reactivex.rxjava3` | `.plugins.RxJavaPlugins;` | `.PublishProcessor;` | `.annotations.GwtIncompatible;` | `= new YearMonthDay(2005, 6, 9, 10, 20, 30, 40);` | `.exceptions.Exceptions;` | `.exceptions.Exceptions;` | `................................` |
| `Binds a proxy listener with the` | `supplied auth mode` | `specified listener.` | `given {@code true}` | `recorded, the timestamp is not-null, it will only be written to the stream.` | `plaintext = "ProxyChannel.localAddress()).getPort();` | `recorded, the recorded, the recorded, the recorded, the recorded, the rec` | `................................` |
| `Type mapType = Utils.getSupertype(type, rawParameterType,` | `Map.class);` | `rawType);` | `"value");` | `true)) {` | `Map.class);` | `"@FieldMap parameter type must be Map.");` | `................................` |
| `assertEquals("p", sel.get(2).tag` | `Name());` | `Name());` | `());` | `);` | `Name());` | `Name());` | `................................` |
| `Parameterized` | `Type wild =` | `Type type =` | `Test` | `Type parameterizedType.getActualTypeArguments()[0];` | `Type) new TypeToken<List<String>>() {}.getType();` | `Type) new TypeCapture<Map<String, String> map = new TypeToken<Map<String, String> map = new TypeToken` | `................................` |
| `protected int` | `cost() {` | `get(int[][] array, int[] array, int[] array, int[] array, int[] array, int[] array, int[] array, int` | `getCaped() {` | `_streamWriteFeatures;` | `_streamWriteFeatures the behavior in {@code (node instanceof Element) {` | `_streamWriteFeatures;` | `................................` |
| `import static com.` | `google.common.truth.Truth.assertThat;` | `google.common.base.Preconditions.checkNotNull;` | `google.common.collect.collect.collect.collect.collect.collect.collect.collect.collect.collect.collect.collect.collect.collect.` | `google.common.collect.testing.features.CollectionSize;` | `google.common.google.common.google.common.google.common.google.common.google.common.google.common.google.common.` | `google.common.google.common.google.common.google.common.google.common.google.common.google.common.google.common.` | `................................` |
| `* @throws Illegal` | `StateException if the next token is not the end of an object.` | `StateException if the current is not initialized` | `StateException if the value is not be null.` | `StateException("No instances!");` | `StateException if the next token from the JSON stream and asserts that it is the end of the current` | `StateException("BEGIN_OBJECT);` | `................................` |
| `import java.lang` | `.reflect.Constructor;` | `.annotation.Retention;` | `.concurrent.concurrent.concurrent.Serializable;` | `.reflect.Method;` | `.jsoup.helpersoup.helpers.jsoup.helpers.jsoup` | `.jsoup.helpersoup.helpers.jsoup.helpers.jsoup` | `................................` |
| `Service` | `MethodAdapterInfo adapterInfo = parseServiceMethodAdapterInfo(method);` | `.getProxyInstance(service);` | `.get(new HashSet(new HashSet<>(new HashSet<>()), "b", "b", "b", "b",` | `.class).start();` | `executor;` | `.getClassLoader(),` | `................................` |
| `Class` | `<?> fieldType = field.getType();` | `<?> cls = cls.getDeclaredFields();` | `<?>[] value = null;` | `Util.throwIfRTE(t);` | `Util.nextName();` | `Util.getTypeDescription(name)) {` | `................................` |
| `public int i` | `;` | `;` | `temy() {` | `;` | `;` | `;` | `................................` |

## Findings

1. **The network beats the required n-gram baseline on every accuracy metric (the n-gram is faster: 1.2 ms against 81 ms).** We compared it with an interpolated modified Kneser-Ney 5-gram built from the same 24.2M train tokens. Top-1 is 49.5% against 35.5% (+14.0 points), top-5 is 70.1% against 57.7%, and perplexity is 12.89 against 32.83 (2.5x lower). Line exact match is 19.4% against 11.4%. On the same 500 lines, the network alone is exact on 52 and the n-gram alone on 12 (exact sign test p = 4.6e-7). The network does this with 4.3 MB of weights, against 168 MB of n-gram tables. The "most-frequent token" baseline scores 4.5% top-1 and 0% exact match.
2. **It also wins like-for-like against a cache model.** We mixed the n-gram with a cache that sees only the tokens the network sees (its 128-token window). That model scores 45.9% top-1, perplexity 18.96 and 14.0% line exact match. The network scores 49.5%, 12.89 and 19.4%.
3. **Negative result: a file-level cache n-gram beats the network on token metrics.** This is the same 5-gram with a cache over up to 2048 earlier tokens of the current file (Tu et al. 2014; Hellendoorn & Devanbu 2017). It reaches 58.2% top-1, 77.7% top-5 and perplexity 9.10. Line exact match is a tie: 19.6% against 19.4%, one line out of 500. On code lines the network is level or slightly ahead (20.7% against 20.5%), and it has the better edit similarity overall (0.427 against 0.403).
   - The class breakdown shows where the gap comes from: identifiers (57.5% against 42.6%) and literals (51.7% against 41.3%). These repeat within a file beyond the network's 128-token window.
   - The network wins on syntax and layout: whitespace 68.0% against 56.8%, punctuation 63.5% against 62.2%.
   - A 1.07M-parameter model with a 128-token context cannot copy an identifier it last saw 500 tokens earlier. A longer context, or a network+cache mixture, is the obvious next step. Neither was tried here.
4. **Compute is what moved the needle.** After 22 minutes (lab-preview, 2.6M tokens), the same architecture sits below the plain 5-gram (29.8% top-1, perplexity 40.9). The release checkpoint, after 130 minutes and 14.0M tokens, is 19.7 points better than the preview and 14.0 points above the n-gram.
5. **Training was step-limited, not data-limited.** The largest stable peak LR was 3e-3; 6e-3 stalled at a 6-minute val loss of 5.84. At 3e-3, shrinking the global batch from 8192 to 4096 to 2048 tokens per step lowered the 6-minute val loss from 5.15 to 4.42 to 4.25, at the same throughput. The release run therefore used 16 workers x 1 sequence (2048 tokens per step at T=128).
6. **jsoup is the hardest test repo for every method.** The network scores 45.4% top-1 and perplexity 17.9 on jsoup, against 52-54% and about 10 on gson and retrofit. A plausible reason, not tested: retrofit and gson have close relatives in the train split (square/okhttp; FasterXML/jackson-*), while jsoup's HTML-parsing code has none.
7. **Comment lines are harder for the network than code lines.** Exact match is 10.8% on comments against 20.7% on code. On comments the network trails the plain 5-gram (12.3%) and both cache variants (13.8%), but n = 65, so the gap is one to two lines. Code lines are where the learned model pays off.
8. **Latency (measured under load).** A rest-of-line completion averages 81 ms (p50 69 ms, p95 168 ms) on one thread with the KV-cached decoder. That includes a prompt prefill of up to 96 tokens and about 9.5 generated tokens. A next-token forward costs 0.65 ms per token in batched windows.
   - The release trainer was no longer running for this measurement. Other jobs on the laptop, outside this project, were busy: the evaluator measured the machine at 98% CPU, with about 14.6 of 16 logical cores used by other processes during the timed phase.
   - Treat these as upper bounds; an idle laptop should be faster. The n-gram answers in 1.2 ms, but its tables take 40x the memory.
9. **Training run and disclosed gap.** The release run had a 150-minute budget, 16 worker threads and 1.07M parameters. Its log survives up to 138.8 min (step 10,195, 14.66M tokens, LR already annealed to 3.4e-4 against a 3e-4 floor).
   - The run never wrote its end-of-run record or last.bin. When the evaluation was redone, the process was gone, and the worktree copy of its output had been removed by a branch reset between worker attempts. The cause and exact time of termination were therefore not observed.
   - The shipped weights.bin (sha256 d91725ce...77b2, recorded in results.json) is the trainer's own best-on-validation EMA checkpoint. It is from step 9,875 at 130.3 min, with val loss 2.5925 after 14.0M tokens (13.1 tokens/param). Only the next validation eval (due at about 140 min) could have replaced it. The table's 138.8 min, 14.66M tokens and 1.76k tok/s describe the whole surviving log.
   - Average throughput (1.76k tok/s, eval time included) is below the 2.66k tok/s of the quiet benchmark, because other jobs shared the CPU during the run.
   - An earlier draft of this report said a harness limit killed the run at 121.3 min. That was wrong: the run continued past 138 min. All numbers here were re-measured on the shipped checkpoint after the trainer had stopped.
10. **Validation and test agree.** The best validation loss (commons-codec, 2.593 nats) is consistent with the test perplexity (ln 12.89 = 2.56 nats). So the checkpoint selected on validation is not overfit to the validation repo.

Reproduce: `npm run evaluate -- --models models/release,models/lab-preview --windows 2700 --lines 500 --order 5 --seed 2026` (10.5 min, one thread). Raw numbers: results/results.json.
