/**
 * WASM SIMD GEMM kernel, assembled byte by byte in TypeScript (no compiler, no wabt, no
 * framework). It computes the same NT product as kernels.ts, C[M,N] (+)= A[M,K] * B[N,K]^T,
 * with f32x4 vectors along K and an R x S register tile of v128 accumulators (2 x 4 by default:
 * 8 accumulators + 2 A-vectors + 1 B-vector fit in the 16 SIMD registers of x86-64).
 *
 * Requirements: K % 4 == 0 (callers fall back to the JS kernel otherwise). Operands are copied
 * into the module's linear memory, the kernel runs, and C is copied back. The copies cost
 * O(MK + NK + MN) against O(MNK) multiply-adds, which is small for the model's shapes.
 */

// ------------------------------------------------------------------ tiny WASM assembler

function uleb(n: number): number[] {
  const out: number[] = [];
  do {
    let b = n & 0x7f;
    n >>>= 7;
    if (n !== 0) b |= 0x80;
    out.push(b);
  } while (n !== 0);
  return out;
}

function sleb(n: number): number[] {
  const out: number[] = [];
  let more = true;
  while (more) {
    let b = n & 0x7f;
    n >>= 7;
    if ((n === 0 && (b & 0x40) === 0) || (n === -1 && (b & 0x40) !== 0)) more = false;
    else b |= 0x80;
    out.push(b);
  }
  return out;
}

const str = (s: string): number[] => [...uleb(s.length), ...Array.from(s, (c) => c.charCodeAt(0))];
const vec = (items: number[][]): number[] => [...uleb(items.length), ...items.flat()];
const section = (id: number, body: number[]): number[] => [id, ...uleb(body.length), ...body];

const I32 = 0x7f, F32 = 0x7d, V128 = 0x7b;
const op = {
  block: [0x02, 0x40], loop: [0x03, 0x40], end: [0x0b], br: 0x0c, brIf: 0x0d,
  localGet: 0x20, localSet: 0x21,
  i32Const: 0x41, i32Add: 0x6a, i32Mul: 0x6c, i32Shl: 0x74, i32GtS: 0x4a, i32GeS: 0x4e,
  f32Load: 0x2a, f32Store: 0x38, f32Const: 0x43, f32Add: 0x92, select: 0x1b,
};
const simd = (code: number, ...imm: number[]): number[] => [0xfd, ...uleb(code), ...imm];
const v128Load = (offset = 0): number[] => simd(0x00, 4, ...uleb(offset));
const f32x4Add = (): number[] => simd(0xe4);
const f32x4Mul = (): number[] => simd(0xe6);
const f32x4ExtractLane = (lane: number): number[] => simd(0x1f, lane);
const v128Zero = (): number[] => simd(0x0c, ...new Array(16).fill(0));

/** Emit the body of tile_RxS(N, K, A, B, C, acc, i0, i1, j0, j1). All pointers are byte offsets. */
function tileFunctionBody(R: number, S: number): number[] {
  // Params 0..9.
  const pN = 0, pK = 1, pA = 2, pB = 3, pC = 4, pAcc = 5, pI0 = 6, pI1 = 7, pJ0 = 8, pJ1 = 9;
  let next = 10;
  const i = next++, j = next++, k = next++, Kb = next++;
  const aRow = Array.from({ length: R }, () => next++);
  const bRow = Array.from({ length: S }, () => next++);
  const cRow = next++;
  const nI32 = next - 10;
  const va = Array.from({ length: R }, () => next++);
  const vb = next++;
  const acc = Array.from({ length: R * S }, () => next++);
  const nV128 = R + 1 + R * S;

  const get = (l: number): number[] => [op.localGet, ...uleb(l)];
  const set = (l: number): number[] => [op.localSet, ...uleb(l)];
  const c32 = (v: number): number[] => [op.i32Const, ...sleb(v)];
  const code: number[] = [];
  const emit = (...xs: (number | number[])[]): void => {
    for (const x of xs) Array.isArray(x) ? code.push(...x) : code.push(x);
  };

  // Kb = K << 2 ; i = i0
  emit(get(pK), c32(2), op.i32Shl, set(Kb));
  emit(get(pI0), set(i));
  emit(op.block, op.loop); // $iExit / $iLoop  (depths: inside loop, br 0 = loop, br 1 = block)
  {
    // if (i + R > i1) break
    emit(get(i), c32(R), op.i32Add, get(pI1), op.i32GtS, op.brIf, 1);
    // aRow0 = A + i*Kb ; aRow_r = aRow_{r-1} + Kb
    emit(get(pA), get(i), get(Kb), op.i32Mul, op.i32Add, set(aRow[0]));
    for (let r = 1; r < R; r++) emit(get(aRow[r - 1]), get(Kb), op.i32Add, set(aRow[r]));
    emit(get(pJ0), set(j));
    emit(op.block, op.loop); // $jExit / $jLoop
    {
      emit(get(j), c32(S), op.i32Add, get(pJ1), op.i32GtS, op.brIf, 1);
      emit(get(pB), get(j), get(Kb), op.i32Mul, op.i32Add, set(bRow[0]));
      for (let s = 1; s < S; s++) emit(get(bRow[s - 1]), get(Kb), op.i32Add, set(bRow[s]));
      for (const a of acc) emit(v128Zero(), set(a));
      emit(c32(0), set(k));
      emit(op.block, op.loop); // $kExit / $kLoop
      {
        emit(get(k), get(Kb), op.i32GeS, op.brIf, 1);
        for (let r = 0; r < R; r++) emit(get(aRow[r]), get(k), op.i32Add, v128Load(), set(va[r]));
        for (let s = 0; s < S; s++) {
          emit(get(bRow[s]), get(k), op.i32Add, v128Load(), set(vb));
          for (let r = 0; r < R; r++) {
            const a = acc[r * S + s];
            emit(get(a), get(va[r]), get(vb), f32x4Mul(), f32x4Add(), set(a));
          }
        }
        emit(get(k), c32(16), op.i32Add, set(k));
        emit(op.br, 0);
      }
      emit(op.end, op.end);
      // Store: C[(i+r)*N + j + s] = hsum(acc) (+ old value if acc flag)
      for (let r = 0; r < R; r++) {
        // cRow = C + ((i + r) * N + j) * 4
        emit(get(pC), get(i), c32(r), op.i32Add, get(pN), op.i32Mul, get(j), op.i32Add, c32(2), op.i32Shl, op.i32Add, set(cRow));
        for (let s = 0; s < S; s++) {
          const a = acc[r * S + s];
          // address for the store
          emit(get(cRow));
          // horizontal sum
          emit(get(a), f32x4ExtractLane(0), get(a), f32x4ExtractLane(1), op.f32Add);
          emit(get(a), f32x4ExtractLane(2), op.f32Add, get(a), f32x4ExtractLane(3), op.f32Add);
          // + (acc ? C[addr] : 0)
          emit(get(cRow), op.f32Load, 2, ...uleb(s * 4), op.f32Const, 0, 0, 0, 0, get(pAcc), op.select, op.f32Add);
          emit(op.f32Store, 2, ...uleb(s * 4));
        }
      }
      emit(get(j), c32(S), op.i32Add, set(j));
      emit(op.br, 0);
    }
    emit(op.end, op.end);
    emit(get(i), c32(R), op.i32Add, set(i));
    emit(op.br, 0);
  }
  emit(op.end, op.end);
  emit(op.end); // function end

  const locals = vec([[...uleb(nI32), I32], [...uleb(nV128), V128]]);
  const body = [...locals, ...code];
  void F32;
  return [...uleb(body.length), ...body];
}

/** Assemble a module exporting tile_2x4 and tile_1x1, importing memory env.mem. */
export function assembleGemmModule(tiles: [number, number][] = [[2, 4], [1, 1]]): Uint8Array<ArrayBuffer> {
  const funcType = [0x60, ...vec(new Array(10).fill([I32])), ...vec([])];
  const typeSec = section(1, vec([funcType]));
  const importSec = section(2, vec([[...str("env"), ...str("mem"), 0x02, 0x00, ...uleb(1)]]));
  const funcSec = section(3, vec(tiles.map(() => [0])));
  const exportSec = section(7, vec(tiles.map(([r, s], idx) => [...str(`tile_${r}x${s}`), 0x00, ...uleb(idx)])));
  const codeSec = section(10, [...uleb(tiles.length), ...tiles.flatMap(([r, s]) => tileFunctionBody(r, s))]);
  return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, ...typeSec, ...importSec, ...funcSec, ...exportSec, ...codeSec]);
}

// ------------------------------------------------------------------ runtime wrapper

type TileFn = (N: number, K: number, A: number, B: number, C: number, acc: number, i0: number, i1: number, j0: number, j1: number) => void;

interface WasmGemm {
  mem: WebAssembly.Memory;
  t24: TileFn;
  t11: TileFn;
}

let inst: WasmGemm | null | undefined;

function instance(): WasmGemm | null {
  if (inst !== undefined) return inst;
  try {
    const mem = new WebAssembly.Memory({ initial: 1 });
    const mod = new WebAssembly.Module(assembleGemmModule());
    const ex = new WebAssembly.Instance(mod, { env: { mem } }).exports as Record<string, TileFn>;
    inst = { mem, t24: ex.tile_2x4, t11: ex.tile_1x1 };
  } catch {
    inst = null; // no WASM SIMD support: callers use the JS kernel
  }
  return inst;
}

export function wasmGemmAvailable(): boolean {
  return instance() !== null;
}

function ensureBytes(w: WasmGemm, bytes: number): void {
  const have = w.mem.buffer.byteLength;
  if (bytes > have) w.mem.grow(Math.ceil((bytes - have) / 65536));
}

const align16 = (n: number): number => (n + 15) & ~15;

/**
 * C[M,N] (+)= A[M,K] * B[N,K]^T via WASM SIMD. Returns false (doing nothing) if the kernel is
 * unavailable or K is not a multiple of 4, so the caller can fall back.
 */
export function gemmNTWasm(
  M: number, N: number, K: number,
  A: Float32Array, aOff: number,
  B: Float32Array, bOff: number,
  C: Float32Array, cOff: number,
  accumulate: boolean,
): boolean {
  if (K % 4 !== 0) return false;
  const w = instance();
  if (!w) return false;
  const aBytes = align16(M * K * 4), bBytes = align16(N * K * 4), cBytes = align16(M * N * 4);
  ensureBytes(w, aBytes + bBytes + cBytes);
  const f = new Float32Array(w.mem.buffer);
  const pA = 0, pB = aBytes, pC = aBytes + bBytes;
  f.set(A.subarray(aOff, aOff + M * K), pA >> 2);
  f.set(B.subarray(bOff, bOff + N * K), pB >> 2);
  if (accumulate) f.set(C.subarray(cOff, cOff + M * N), pC >> 2);
  const acc = accumulate ? 1 : 0;
  const M2 = M - (M % 2), N4 = N - (N % 4);
  // Column blocking: keep a ~64 KB panel of B rows hot in L2 while all rows of A stream past it.
  const jb = Math.max(4, Math.floor(16384 / K) & ~3);
  for (let j0 = 0; j0 < N4; j0 += jb) w.t24(N, K, pA, pB, pC, acc, 0, M2, j0, Math.min(N4, j0 + jb));
  if (N4 < N) w.t11(N, K, pA, pB, pC, acc, 0, M2, N4, N);
  if (M2 < M) w.t11(N, K, pA, pB, pC, acc, M2, M, 0, N);
  C.set(f.subarray(pC >> 2, (pC >> 2) + M * N), cOff);
  return true;
}
