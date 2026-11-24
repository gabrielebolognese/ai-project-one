/**
 * Lab-notebook data for the web app: everything the training and evaluation scripts wrote to disk.
 *
 *   runs[]      models/<run>/{config.json, summary.json, log.jsonl} (every run, newest first)
 *   results     results/results.json (evaluation vs baselines), or null if `npm run evaluate` has not run
 *   resultsMd   results/results.md, or null
 *   bench       results/bench.json (model-size decision), corpus = results/corpus-stats.json
 *   modelCard   models/<loaded>/MODEL_CARD.md or models/MODEL_CARD.md, else null (the web app then
 *               renders a card from config + measured numbers)
 *
 * Nothing here is synthesised: missing files are reported as null so the UI can say "not run yet".
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import type { ModelHost } from "./model-host.js";

function readJson(path: string): unknown {
  if (!existsSync(path)) return null;
  try { return JSON.parse(readFileSync(path, "utf8")); } catch { return null; }
}

function readText(path: string): string | null {
  return existsSync(path) ? readFileSync(path, "utf8") : null;
}

function readLog(path: string, maxPoints = 2000): unknown[] {
  if (!existsSync(path)) return [];
  const rows: unknown[] = [];
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const l = line.trim();
    if (!l) continue;
    try { rows.push(JSON.parse(l)); } catch { /* a partially written last line while training */ }
  }
  if (rows.length <= maxPoints) return rows;
  // Thin evenly but keep every eval row (they carry val_loss).
  const stride = Math.ceil(rows.length / maxPoints);
  return rows.filter((r, i) => i % stride === 0 || (r as { val_loss?: number }).val_loss !== undefined || i === rows.length - 1);
}

export function readNotebook(rootDir: string, host: ModelHost) {
  const root = resolve(rootDir);
  const modelsDir = join(root, "models");
  const runs: { name: string; mtime: number; loaded: boolean; config: unknown; summary: unknown; log: unknown[] }[] = [];
  const loadedDir = host.source.kind === "random" ? null : resolve(host.source.dir);
  if (existsSync(modelsDir)) {
    for (const e of readdirSync(modelsDir, { withFileTypes: true })) {
      if (!e.isDirectory()) continue;
      const dir = join(modelsDir, e.name);
      const logPath = join(dir, "log.jsonl");
      if (!existsSync(logPath) && !existsSync(join(dir, "summary.json"))) continue;
      runs.push({
        name: e.name,
        mtime: existsSync(logPath) ? statSync(logPath).mtimeMs : statSync(dir).mtimeMs,
        loaded: loadedDir === resolve(dir),
        config: readJson(join(dir, "config.json")),
        summary: readJson(join(dir, "summary.json")),
        log: readLog(logPath),
      });
    }
  }
  runs.sort((a, b) => Number(b.loaded) - Number(a.loaded) || b.mtime - a.mtime);
  const resultsDir = join(root, "results");
  const cardCandidates = [loadedDir ? join(loadedDir, "MODEL_CARD.md") : "", join(modelsDir, "MODEL_CARD.md")].filter(Boolean);
  const cardPath = cardCandidates.find((p) => existsSync(p));
  return {
    model: host.modelInfo().sourceLabel,
    source: host.source,
    runs,
    results: readJson(join(resultsDir, "results.json")),
    resultsMd: readText(join(resultsDir, "results.md")),
    bench: readJson(join(resultsDir, "bench.json")),
    corpus: readJson(join(resultsDir, "corpus-stats.json")),
    webPerf: readJson(join(resultsDir, "web-perf.json")),
    modelCard: cardPath ? readText(cardPath) : null,
  };
}
