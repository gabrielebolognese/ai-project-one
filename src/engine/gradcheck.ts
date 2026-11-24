/**
 * Finite-difference gradient checker used by the tests.
 *
 * The scalar objective is L = sum(f(inputs) * R) for a fixed random R, so every output element
 * contributes a different weight. Analytic gradients come from the tape; numeric ones from
 * central differences evaluated in inference mode with float64 accumulation of L.
 */
import { Tensor, backward, clearTape, noGrad } from "./tensor.js";
import { mul, sum } from "./ops.js";
import { Rng } from "./rng.js";

export interface GradCheckOptions {
  eps?: number;
  atol?: number;
  rtol?: number;
  /** Check at most this many elements per input (random subset). Default: all. */
  maxPerInput?: number;
  seed?: number;
}

export interface GradCheckReport {
  ok: boolean;
  checked: number;
  worst: { input: number; index: number; analytic: number; numeric: number; err: number } | null;
  failures: string[];
}

export function gradCheck(fn: (inputs: Tensor[]) => Tensor, inputs: Tensor[], opts: GradCheckOptions = {}): GradCheckReport {
  const eps = opts.eps ?? 1e-2;
  const atol = opts.atol ?? 2e-3;
  const rtol = opts.rtol ?? 2e-2;
  const rng = new Rng(opts.seed ?? 7);

  clearTape();
  for (const t of inputs) {
    t.requiresGrad = true;
    t.grad = new Float32Array(t.size);
  }
  const out = fn(inputs);
  const R = new Tensor(new Float32Array(out.size), out.shape);
  for (let i = 0; i < R.size; i++) R.data[i] = rng.next() * 2 - 1;
  const loss = sum(mul(out, R));
  backward(loss);
  const analytic = inputs.map((t) => Float64Array.from(t.grad ?? new Float32Array(t.size)));

  const objective = (): number =>
    noGrad(() => {
      const o = fn(inputs);
      let s = 0;
      for (let i = 0; i < o.size; i++) s += o.data[i] * R.data[i];
      return s;
    });

  const failures: string[] = [];
  let checked = 0;
  let worst: GradCheckReport["worst"] = null;
  inputs.forEach((t, ti) => {
    let idx = [...Array(t.size).keys()];
    if (opts.maxPerInput !== undefined && idx.length > opts.maxPerInput) {
      for (let i = idx.length - 1; i > 0; i--) {
        const j = rng.int(i + 1);
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      idx = idx.slice(0, opts.maxPerInput);
    }
    for (const i of idx) {
      const orig = t.data[i];
      t.data[i] = orig + eps;
      const lp = objective();
      t.data[i] = orig - eps;
      const lm = objective();
      t.data[i] = orig;
      const num = (lp - lm) / (2 * eps);
      const ana = analytic[ti][i];
      const err = Math.abs(num - ana);
      const tol = atol + rtol * Math.max(Math.abs(num), Math.abs(ana));
      checked++;
      if (!worst || err > worst.err) worst = { input: ti, index: i, analytic: ana, numeric: num, err };
      if (err > tol) failures.push(`input ${ti}${t.name ? ` (${t.name})` : ""}[${i}]: analytic ${ana.toExponential(4)} vs numeric ${num.toExponential(4)}`);
    }
  });
  return { ok: failures.length === 0, checked, worst, failures };
}
