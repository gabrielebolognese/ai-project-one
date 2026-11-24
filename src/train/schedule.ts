/**
 * Budget-aware schedules. Progress is rho = elapsed / budget in [0, 1]; because every schedule is
 * a function of rho (not of a step count fixed in advance) any wall-clock budget ends with a fully
 * annealed model. Same idea as TinyMind's lr_at.
 */

export interface LrSchedule {
  peakLr: number;
  /** Final LR as a fraction of the peak (cosine floor). */
  minLrFrac: number;
  /** Warmup ends after this many steps or this fraction of the budget, whichever comes first. */
  warmupSteps: number;
  warmupRho: number;
}

export function lrAt(s: LrSchedule, step: number, rho: number): number {
  const r = Math.min(Math.max(rho, 0), 1);
  const warm = Math.min(1, Math.max((step + 1) / Math.max(1, s.warmupSteps), r / Math.max(1e-9, s.warmupRho)));
  const cos = 0.5 * (1 + Math.cos(Math.PI * r));
  return s.peakLr * warm * (s.minLrFrac + (1 - s.minLrFrac) * cos);
}

export interface CtxCurriculum {
  enabled: boolean;
  /** Context used before `switchRho`. */
  startCtx: number;
  fullCtx: number;
  switchRho: number;
}

/** Short windows early (cheaper steps), the full context after switchRho. */
export function ctxAt(c: CtxCurriculum, rho: number): number {
  return c.enabled && rho < c.switchRho ? c.startCtx : c.fullCtx;
}

/** EMA decay with a warm start so early EMA weights are not stuck near the initialisation. */
export function emaDecayAt(target: number, step: number): number {
  return Math.min(target, (1 + step) / (10 + step));
}
