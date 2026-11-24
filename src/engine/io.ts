/**
 * Checkpoint format (read by the trainer, the inference server and the web lab):
 *
 *   models/<run>/weights.bin   raw little-endian float32, all tensors concatenated in order
 *   models/<run>/config.json   { ..., "tensors": [{ name, shape, offset, numel, dtype: "f32" }] }
 *
 * `offset` and `numel` are in floats, so a browser can do
 *   new Float32Array(await (await fetch("weights.bin")).arrayBuffer(), offset * 4, numel)
 * with zero copying. Every tensor the lab lists comes straight from this table.
 */
import { mkdirSync, readFileSync, writeFileSync, renameSync } from "node:fs";
import { dirname, join } from "node:path";
import type { ParamStore } from "./params.js";

export interface TensorInfo {
  name: string;
  shape: number[];
  offset: number;
  numel: number;
  dtype: "f32";
}

export function tensorTable(store: ParamStore): TensorInfo[] {
  return store.entries.map((e) => ({ name: e.name, shape: [...e.shape], offset: e.offset, numel: e.numel, dtype: "f32" }));
}

function isLittleEndian(): boolean {
  return new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
}

/** Atomic write (tmp + rename) so a crash never leaves a half-written checkpoint. */
function writeAtomic(path: string, data: string | Uint8Array): void {
  mkdirSync(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, data);
  renameSync(tmp, path);
}

/** Write a flat float32 buffer as raw little-endian bytes. */
export function writeFloat32(path: string, data: Float32Array): void {
  if (!isLittleEndian()) throw new Error("big-endian hosts are not supported");
  // Copy so a SharedArrayBuffer-backed view can be written safely.
  const copy = new Float32Array(data);
  writeAtomic(path, new Uint8Array(copy.buffer));
}

export function readFloat32(path: string): Float32Array {
  const buf = readFileSync(path);
  if (buf.byteLength % 4 !== 0) throw new Error(`${path}: size ${buf.byteLength} is not a multiple of 4`);
  const out = new Float32Array(buf.byteLength / 4);
  new Uint8Array(out.buffer).set(buf);
  return out;
}

export function writeJson(path: string, obj: unknown): void {
  writeAtomic(path, JSON.stringify(obj, null, 2) + "\n");
}

/**
 * Save weights + config. `config` is any JSON-serialisable model description; the tensor table
 * is added under "tensors".
 */
export function saveCheckpoint(
  dir: string,
  store: ParamStore,
  config: Record<string, unknown>,
  opts: { weightsFile?: string; data?: Float32Array } = {},
): void {
  const weightsFile = opts.weightsFile ?? "weights.bin";
  const data = opts.data ?? store.data;
  if (data.length !== store.size) throw new Error("saveCheckpoint: data size mismatch");
  writeFloat32(join(dir, weightsFile), data);
  writeJson(join(dir, "config.json"), { ...config, weightsFile, numParams: store.size, tensors: tensorTable(store) });
}

export interface LoadedCheckpoint<C> {
  config: C & { tensors: TensorInfo[] };
  data: Float32Array;
}

/** Load config.json + a weights file and verify every tensor fits. */
export function loadCheckpoint<C = Record<string, unknown>>(dir: string, weightsFile = "weights.bin"): LoadedCheckpoint<C> {
  const config = JSON.parse(readFileSync(join(dir, "config.json"), "utf8")) as C & { tensors: TensorInfo[] };
  const data = readFloat32(join(dir, weightsFile));
  for (const t of config.tensors) {
    if (t.offset + t.numel > data.length) throw new Error(`${weightsFile}: tensor ${t.name} out of range`);
  }
  return { config, data };
}

/** Copy loaded weights into a ParamStore, matching tensors by name and checking shapes. */
export function loadIntoStore(store: ParamStore, tensors: TensorInfo[], data: Float32Array): void {
  const byName = new Map(tensors.map((t) => [t.name, t]));
  for (const e of store.entries) {
    const t = byName.get(e.name);
    if (!t) throw new Error(`checkpoint is missing tensor ${e.name}`);
    if (t.numel !== e.numel || t.shape.join(",") !== e.shape.join(",")) {
      throw new Error(`shape mismatch for ${e.name}: checkpoint [${t.shape}] vs model [${e.shape}]`);
    }
    store.data.set(data.subarray(t.offset, t.offset + t.numel), e.offset);
  }
}
