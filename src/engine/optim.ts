/** AdamW (decoupled weight decay, Loshchilov & Hutter 1711.05101) and global-norm clipping. */

export interface AdamWOptions {
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
}

export class AdamW {
  readonly m: Float32Array;
  readonly v: Float32Array;
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  step = 0;

  constructor(
    readonly size: number,
    /** 1 = apply weight decay to this element. */
    readonly decayMask: Uint8Array | null,
    opts: AdamWOptions = {},
  ) {
    this.m = new Float32Array(size);
    this.v = new Float32Array(size);
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.95;
    this.eps = opts.eps ?? 1e-8;
    this.weightDecay = opts.weightDecay ?? 0.1;
  }

  /** One update of `params` in place from `grad` at learning rate `lr`. */
  update(params: Float32Array, grad: Float32Array, lr: number): void {
    this.step++;
    const { m, v, beta1, beta2, eps, decayMask } = this;
    const bc1 = 1 - Math.pow(beta1, this.step);
    const bc2 = 1 - Math.pow(beta2, this.step);
    const stepSize = lr / bc1;
    const invBc2 = 1 / bc2;
    const wdFactor = 1 - lr * this.weightDecay;
    const n = this.size;
    for (let i = 0; i < n; i++) {
      const g = grad[i];
      const mi = beta1 * m[i] + (1 - beta1) * g;
      const vi = beta2 * v[i] + (1 - beta2) * g * g;
      m[i] = mi;
      v[i] = vi;
      let p = params[i];
      if (decayMask === null || decayMask[i] === 1) p *= wdFactor;
      params[i] = p - (stepSize * mi) / (Math.sqrt(vi * invBc2) + eps);
    }
  }
}

/** L2 norm of a gradient buffer. */
export function gradNorm(grad: Float32Array): number {
  let s = 0;
  for (let i = 0; i < grad.length; i++) s += grad[i] * grad[i];
  return Math.sqrt(s);
}

/** Scale `grad` in place so its global L2 norm is at most maxNorm. Returns the pre-clip norm. */
export function clipGradNorm(grad: Float32Array, maxNorm: number): number {
  const norm = gradNorm(grad);
  if (norm > maxNorm && norm > 0) {
    const s = maxNorm / norm;
    for (let i = 0; i < grad.length; i++) grad[i] *= s;
  }
  return norm;
}
