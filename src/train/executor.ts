/**
 * Data-parallel execution of forward/backward passes.
 *
 * The weights live in one SharedArrayBuffer. Each replica (a worker thread, or the main thread
 * for `LocalExecutor`) builds a Transformer whose parameter tensors are views into that buffer
 * and whose gradients are views into the replica's own shared gradient slab. A training step
 * sends each replica a list of window offsets; replicas compute the mean loss on their
 * micro-batch and leave d(loss)/d(params) in their slab. The trainer then averages the slabs
 * (an all-reduce in shared memory, about P*W adds) and runs AdamW on the shared weights while the
 * replicas are idle, so no locks are needed.
 */
import { Worker } from "node:worker_threads";
import { backward, clearTape, noGrad } from "../engine/index.js";
import { Transformer, type ModelConfig } from "../model/index.js";
import { buildBatch } from "./data.js";

export interface StepResult {
  /** Mean loss per replica. */
  losses: number[];
}

export interface EvalResult {
  lossSum: number;
  count: number;
}

export interface Executor {
  readonly replicas: number;
  /** Per-replica gradient buffers (same layout as the param buffer). */
  readonly grads: Float32Array[];
  step(offsets: number[][], T: number): Promise<StepResult>;
  evaluate(offsets: number[], T: number): Promise<EvalResult>;
  close(): Promise<void>;
}

export interface ExecutorInit {
  cfg: ModelConfig;
  params: Float32Array; // SharedArrayBuffer-backed
  train: Uint16Array; // SharedArrayBuffer-backed
  val: Uint16Array; // SharedArrayBuffer-backed
}

// ------------------------------------------------------------------------------------------------
// Replica logic shared by the worker thread and the in-process executor.
// ------------------------------------------------------------------------------------------------

export class Replica {
  readonly model: Transformer;

  constructor(cfg: ModelConfig, params: Float32Array, grad: Float32Array, private train: Uint16Array, private val: Uint16Array) {
    this.model = new Transformer(cfg, { data: params, grad });
  }

  step(offsets: number[], T: number): number {
    const { ids, targets } = buildBatch(this.train, offsets, T);
    this.model.store.zeroGrad();
    clearTape();
    const { loss } = this.model.forward(ids, offsets.length, T, { targets });
    backward(loss!);
    return loss!.data[0];
  }

  evaluate(offsets: number[], T: number, chunk = 8): EvalResult {
    let lossSum = 0, count = 0;
    noGrad(() => {
      for (let i = 0; i < offsets.length; i += chunk) {
        const part = offsets.slice(i, i + chunk);
        const { ids, targets } = buildBatch(this.val, part, T);
        const { loss } = this.model.forward(ids, part.length, T, { targets });
        lossSum += loss!.data[0] * part.length * T;
        count += part.length * T;
      }
    });
    return { lossSum, count };
  }
}

// ------------------------------------------------------------------------------------------------
// In-process executor (one replica on the main thread): tests and debugging.
// ------------------------------------------------------------------------------------------------

export class LocalExecutor implements Executor {
  readonly replicas = 1;
  readonly grads: Float32Array[];
  private replica: Replica;

  constructor(init: ExecutorInit) {
    const g = new Float32Array(init.params.length);
    this.grads = [g];
    this.replica = new Replica(init.cfg, init.params, g, init.train, init.val);
  }

  async step(offsets: number[][], T: number): Promise<StepResult> {
    return { losses: [this.replica.step(offsets[0], T)] };
  }

  async evaluate(offsets: number[], T: number): Promise<EvalResult> {
    return this.replica.evaluate(offsets, T);
  }

  async close(): Promise<void> {}
}

// ------------------------------------------------------------------------------------------------
// Worker-thread pool.
// ------------------------------------------------------------------------------------------------

export type WorkerRequest =
  | { type: "step"; id: number; offsets: number[]; T: number }
  | { type: "eval"; id: number; offsets: number[]; T: number };

export type WorkerResponse =
  | { type: "ready" }
  | { type: "step"; id: number; loss: number }
  | { type: "eval"; id: number; lossSum: number; count: number }
  | { type: "error"; id: number; message: string };

export interface WorkerInitData {
  cfg: ModelConfig;
  params: Float32Array;
  grad: Float32Array;
  train: Uint16Array;
  val: Uint16Array;
}

function createWorker(data: WorkerInitData): Worker {
  // Under tsx the sources are .ts. Worker threads do not inherit loaders registered with
  // module.register, so a tiny eval'd bootstrap registers tsx in the worker and then imports the
  // TypeScript entry. Compiled output (.js) is started directly.
  if (import.meta.url.endsWith(".ts")) {
    const api = import.meta.resolve("tsx/esm/api");
    const entry = new URL("./worker.ts", import.meta.url).href;
    const boot = `import(${JSON.stringify(api)}).then((m) => { m.register(); return import(${JSON.stringify(entry)}); });`;
    return new Worker(boot, { eval: true, workerData: data });
  }
  return new Worker(new URL("./worker.js", import.meta.url), { workerData: data });
}

interface Pending {
  resolve: (r: WorkerResponse) => void;
  reject: (e: Error) => void;
}

export class WorkerPool implements Executor {
  readonly grads: Float32Array[] = [];
  private workers: Worker[] = [];
  private pending = new Map<number, Pending>();
  private nextId = 1;

  private constructor(readonly replicas: number) {}

  static async create(init: ExecutorInit, replicas: number): Promise<WorkerPool> {
    const pool = new WorkerPool(replicas);
    const ready: Promise<void>[] = [];
    for (let w = 0; w < replicas; w++) {
      const grad = new Float32Array(new SharedArrayBuffer(init.params.length * 4));
      pool.grads.push(grad);
      const data: WorkerInitData = { cfg: init.cfg, params: init.params, grad, train: init.train, val: init.val };
      const worker = createWorker(data);
      pool.workers.push(worker);
      ready.push(
        new Promise<void>((resolve, reject) => {
          const onMsg = (m: WorkerResponse): void => {
            if (m.type === "ready") {
              worker.off("message", onMsg);
              resolve();
            }
          };
          worker.on("message", onMsg);
          worker.once("error", reject);
        }),
      );
      worker.on("message", (m: WorkerResponse) => {
        if (m.type === "ready") return;
        const p = pool.pending.get(m.id);
        if (!p) return;
        pool.pending.delete(m.id);
        if (m.type === "error") p.reject(new Error(`worker ${w}: ${m.message}`));
        else p.resolve(m);
      });
      worker.on("error", (e) => {
        for (const p of pool.pending.values()) p.reject(e instanceof Error ? e : new Error(String(e)));
        pool.pending.clear();
      });
    }
    await Promise.all(ready);
    return pool;
  }

  private call(w: number, req: Omit<WorkerRequest, "id">): Promise<WorkerResponse> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.workers[w].postMessage({ ...req, id });
    });
  }

  async step(offsets: number[][], T: number): Promise<StepResult> {
    if (offsets.length !== this.replicas) throw new Error("step: one offset list per replica");
    const res = await Promise.all(offsets.map((o, w) => this.call(w, { type: "step", offsets: o, T })));
    return { losses: res.map((r) => (r.type === "step" ? r.loss : NaN)) };
  }

  async evaluate(offsets: number[], T: number): Promise<EvalResult> {
    const parts: number[][] = Array.from({ length: this.replicas }, () => []);
    offsets.forEach((o, i) => parts[i % this.replicas].push(o));
    const res = await Promise.all(
      parts.map((p, w) => (p.length ? this.call(w, { type: "eval", offsets: p, T }) : Promise.resolve(null))),
    );
    let lossSum = 0, count = 0;
    for (const r of res) if (r && r.type === "eval") { lossSum += r.lossSum; count += r.count; }
    return { lossSum, count };
  }

  async close(): Promise<void> {
    await Promise.all(this.workers.map((w) => w.terminate()));
    this.workers = [];
  }
}

/** Average the replicas' gradients into `out` (out = mean_w grads[w]). */
export function averageGrads(grads: Float32Array[], out: Float32Array): void {
  const W = grads.length;
  out.set(grads[0]);
  for (let w = 1; w < W; w++) {
    const g = grads[w];
    for (let i = 0; i < out.length; i++) out[i] += g[i];
  }
  if (W > 1) {
    const s = 1 / W;
    for (let i = 0; i < out.length; i++) out[i] *= s;
  }
}
