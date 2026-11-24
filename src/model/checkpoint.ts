/** Save / load a Transformer as models/<run>/{config.json, weights.bin}. */
import { loadCheckpoint, loadIntoStore, saveCheckpoint, type TensorInfo } from "../engine/index.js";
import type { ModelConfig } from "./config.js";
import { Transformer } from "./transformer.js";

export interface SavedModelConfig {
  format: "jal-v1";
  model: ModelConfig;
  [extra: string]: unknown;
}

export function saveModel(dir: string, model: Transformer, extra: Record<string, unknown> = {}, opts: { weightsFile?: string; data?: Float32Array } = {}): void {
  const config: SavedModelConfig = { format: "jal-v1", model: model.cfg, ...extra };
  saveCheckpoint(dir, model.store, config, opts);
}

export function loadModel(dir: string, weightsFile = "weights.bin"): { model: Transformer; config: SavedModelConfig & { tensors: TensorInfo[] } } {
  const { config, data } = loadCheckpoint<SavedModelConfig>(dir, weightsFile);
  if (config.format !== "jal-v1") throw new Error(`${dir}: unknown checkpoint format ${String(config.format)}`);
  const model = new Transformer(config.model);
  loadIntoStore(model.store, config.tensors, data);
  return { model, config };
}
