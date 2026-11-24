/**
 * ParamStore: every parameter is a view into ONE flat Float32Array (and its gradient into one flat
 * gradient array). This makes AdamW, global-norm clipping, EMA, checkpointing and the
 * data-parallel gradient all-reduce simple loops over a single buffer, and lets worker threads
 * share the weights through a SharedArrayBuffer with zero copies.
 */
import { Tensor, numel, type Shape } from "./tensor.js";

export interface ParamSpec {
  name: string;
  shape: number[];
  /** Apply weight decay (true for matrices, false for biases / norms / position tables). */
  decay: boolean;
}

export interface ParamEntry extends ParamSpec {
  offset: number;
  numel: number;
  tensor: Tensor;
}

export interface ParamStoreOptions {
  /** Existing buffer for the weights (e.g. a SharedArrayBuffer view). Allocated if absent. */
  data?: Float32Array;
  /** Existing buffer for the gradients. Allocated if absent. */
  grad?: Float32Array;
  /** Allocate the weight buffer on a SharedArrayBuffer. */
  shared?: boolean;
}

export class ParamStore {
  readonly entries: ParamEntry[] = [];
  readonly byName = new Map<string, ParamEntry>();
  readonly data: Float32Array;
  readonly grad: Float32Array;
  readonly size: number;

  constructor(specs: ParamSpec[], opts: ParamStoreOptions = {}) {
    let total = 0;
    for (const s of specs) total += numel(s.shape);
    this.size = total;
    if (opts.data) {
      if (opts.data.length !== total) throw new Error(`param buffer has ${opts.data.length} floats, need ${total}`);
      this.data = opts.data;
    } else {
      this.data = opts.shared ? new Float32Array(new SharedArrayBuffer(total * 4)) : new Float32Array(total);
    }
    if (opts.grad) {
      if (opts.grad.length !== total) throw new Error(`grad buffer has ${opts.grad.length} floats, need ${total}`);
      this.grad = opts.grad;
    } else {
      this.grad = new Float32Array(total);
    }
    let off = 0;
    for (const s of specs) {
      const n = numel(s.shape);
      const t = new Tensor(this.data.subarray(off, off + n), s.shape, true, s.name);
      t.grad = this.grad.subarray(off, off + n);
      const e: ParamEntry = { ...s, shape: [...s.shape], offset: off, numel: n, tensor: t };
      this.entries.push(e);
      if (this.byName.has(s.name)) throw new Error(`duplicate param name ${s.name}`);
      this.byName.set(s.name, e);
      off += n;
    }
  }

  get(name: string): Tensor {
    const e = this.byName.get(name);
    if (!e) throw new Error(`unknown param ${name}`);
    return e.tensor;
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  /** Boolean mask (1 = decay) per element, used by AdamW. */
  decayMask(): Uint8Array {
    const m = new Uint8Array(this.size);
    for (const e of this.entries) if (e.decay) m.fill(1, e.offset, e.offset + e.numel);
    return m;
  }

  specs(): ParamSpec[] {
    return this.entries.map((e) => ({ name: e.name, shape: [...e.shape], decay: e.decay }));
  }
}

export function shapeEq(a: Shape, b: Shape): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}
