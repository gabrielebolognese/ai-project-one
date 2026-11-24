/** Model hyper-parameters and named size presets (benchmarked in scripts/bench.ts). */
export interface ModelConfig {
  vocabSize: number;
  /** Maximum context length (size of the learned position table). */
  ctx: number;
  dModel: number;
  nLayers: number;
  nHeads: number;
  /** MLP hidden width (4 * dModel by default). */
  dMlp: number;
}

export type PresetName = "small" | "base" | "large";

export const PRESETS: Record<PresetName, Omit<ModelConfig, "vocabSize">> = {
  small: { ctx: 128, dModel: 96, nLayers: 3, nHeads: 3, dMlp: 384 },
  base: { ctx: 128, dModel: 128, nLayers: 4, nHeads: 4, dMlp: 512 },
  large: { ctx: 128, dModel: 160, nLayers: 6, nHeads: 5, dMlp: 640 },
};

export function makeConfig(preset: PresetName, vocabSize: number, overrides: Partial<ModelConfig> = {}): ModelConfig {
  const cfg: ModelConfig = { vocabSize, ...PRESETS[preset], ...overrides };
  validateConfig(cfg);
  return cfg;
}

export function validateConfig(cfg: ModelConfig): void {
  for (const [k, v] of Object.entries(cfg)) {
    if (!Number.isInteger(v) || v <= 0) throw new Error(`model config: ${k} must be a positive integer (got ${v})`);
  }
  if (cfg.dModel % cfg.nHeads !== 0) throw new Error("model config: dModel must be divisible by nHeads");
}

/** Parameter count without building the model. */
export function countParams(cfg: ModelConfig): number {
  const { vocabSize: V, ctx: T, dModel: d, nLayers: L, dMlp: m } = cfg;
  const block = 2 * d /*ln1*/ + 3 * d * d + 3 * d /*qkv*/ + d * d + d /*proj*/ + 2 * d /*ln2*/ + m * d + m + d * m + d;
  return V * d + T * d + L * block + 2 * d;
}

/** Approximate training FLOPs per token: 6 * (non-embedding params + tied head) + attention. */
export function trainFlopsPerToken(cfg: ModelConfig, T = cfg.ctx): number {
  const { vocabSize: V, dModel: d, nLayers: L, dMlp: m } = cfg;
  const matmulParams = L * (4 * d * d + 2 * d * m) + V * d; // per-token matmul weights (head included)
  const attn = L * 2 * T * d; // QK^T and PV, averaged over causal positions (~T/2 each, x2 for both)
  return 6 * matmulParams + 3 * attn;
}
