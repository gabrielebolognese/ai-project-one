/**
 * Tensor + tape-based reverse-mode autograd.
 *
 * A Tensor is a contiguous row-major Float32Array with a shape. When gradient recording is on and
 * any input requires a gradient, each op pushes a backward closure onto the thread-local tape.
 * `backward(loss)` seeds d(loss)/d(loss) = 1 and runs the closures in reverse creation order,
 * which is a valid reverse topological order because every op only reads earlier tensors.
 *
 * Parameters normally own a gradient view into one flat gradient buffer (see ParamStore), so the
 * optimiser, gradient clipping and the data-parallel all-reduce work on a single Float32Array.
 */

export type Shape = readonly number[];

export function numel(shape: Shape): number {
  let n = 1;
  for (const s of shape) n *= s;
  return n;
}

export class Tensor {
  data: Float32Array;
  shape: number[];
  grad: Float32Array | null = null;
  requiresGrad: boolean;
  name: string;

  constructor(data: Float32Array, shape: Shape, requiresGrad = false, name = "") {
    if (data.length !== numel(shape)) {
      throw new Error(`Tensor data length ${data.length} does not match shape [${shape.join(",")}]`);
    }
    this.data = data;
    this.shape = [...shape];
    this.requiresGrad = requiresGrad;
    this.name = name;
  }

  static zeros(shape: Shape, requiresGrad = false, name = ""): Tensor {
    return new Tensor(new Float32Array(numel(shape)), shape, requiresGrad, name);
  }

  static from(values: ArrayLike<number>, shape: Shape, requiresGrad = false, name = ""): Tensor {
    return new Tensor(Float32Array.from(values), shape, requiresGrad, name);
  }

  get size(): number {
    return this.data.length;
  }

  /** Gradient buffer, allocated (zeroed) on first use. */
  g(): Float32Array {
    if (this.grad === null) this.grad = new Float32Array(this.data.length);
    return this.grad;
  }

  zeroGrad(): void {
    if (this.grad) this.grad.fill(0);
  }

  /** Reshape without copying (shares data and grad). Gradients flow through the shared buffers. */
  view(shape: Shape): Tensor {
    const t = new Tensor(this.data, shape, this.requiresGrad, this.name);
    if (this.requiresGrad) {
      // Share the gradient buffer so writes to the view's grad land in the source.
      t.grad = this.g();
    }
    return t;
  }
}

type BackwardFn = () => void;

let tape: BackwardFn[] = [];
let gradEnabled = true;

export function isGradEnabled(): boolean {
  return gradEnabled;
}

/** Run `fn` without recording any backward closures (inference mode). */
export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled;
  gradEnabled = false;
  try {
    return fn();
  } finally {
    gradEnabled = prev;
  }
}

/** True when an op with these inputs must record a backward closure. */
export function needsGrad(...inputs: (Tensor | null | undefined)[]): boolean {
  if (!gradEnabled) return false;
  for (const t of inputs) if (t && t.requiresGrad) return true;
  return false;
}

export function record(fn: BackwardFn): void {
  tape.push(fn);
}

export function tapeLength(): number {
  return tape.length;
}

/** Discard the tape without running it (e.g. after an eval forward pass with grad on). */
export function clearTape(): void {
  tape = [];
}

/** Backpropagate from a scalar loss. Accumulates into existing .grad buffers. Clears the tape. */
export function backward(loss: Tensor): void {
  if (loss.size !== 1) throw new Error("backward() expects a scalar loss");
  loss.g()[0] += 1;
  const t = tape;
  tape = [];
  for (let i = t.length - 1; i >= 0; i--) t[i]();
}
