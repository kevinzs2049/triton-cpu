# SVE2 i8mm Notes (Triton CPU)

This note summarizes the SVE2 i8mm dot lowering work, how to benchmark it,
and how to validate correctness.

## Summary of key changes

- SVE2 i8mm lowering now uses 64-bit lane packing to match the llama.cpp
  zip1/zip2 semantics and reduce element-wise insert/extract overhead.
- pack/unpack for 2x2 i32 tiles now avoids per-element ops by using
  `bitcast` + `shape_cast` and 64-bit lane insert/extract.
- 4x4 accumulator assembly now uses `vector::InsertStridedSliceOp` instead
  of per-element inserts.
- B side now uses a local transpose + row-pack strategy (instead of
  extracting columns), reducing slice/extract IR and aligning with
  `smmla`'s 2x8x8x2 semantics.
- The lowering reorders loops to pre-pack B once per `n` tile (across all
  `m` tiles), amortizing B packing cost and improving reuse.

## Optimization principles (why it helps)

1) **Align the pack layout with `smmla` semantics**  
   `smmla` operates on 2x8 and 8x2 fragments per 128-bit segment. By
   transposing B tiles locally (e.g., 16x2 → 2x16 or 8x2 → 2x8) and packing
   rows with `zip1/zip2`, the values land in the exact lane order expected
   by `smmla`, avoiding extra permutes.

2) **Avoid per-element insert/extract**  
   Element-wise `vector::ExtractElement` / `InsertElement` is expensive in
   IR and codegen. Packing through 64-bit lane `bitcast` + 2 inserts
   collapses dozens of ops into a small, predictable sequence.

3) **Amortize B packing across M**  
   For a given `n` tile and `k` step, B does not depend on `m`. By hoisting
   B packing outside the `m` loop, we pay the pack cost once and reuse it
   for all `m` tiles, which boosts throughput for large M.

4) **Keep A/B access contiguous where possible**  
   The local transpose approach on B and row-pack on A keeps loads
   contiguous, reducing strided slice pressure and helping vectorized
   memory paths.

## Dataflow sketch

```
For each n tile:
  prepack B (all k-steps) once:
    B_tile (Kx2) -> local transpose -> row-pack -> zip -> B_vec

For each m tile:
  for each k-step:
    A_tile (2xK) -> row-pack -> zip -> A_vec
    acc = smmla(acc, A_vec, B_vec)
  store acc
```

### Lane packing picture (K%16 path)

```
B 16x2 (col0, col1)         transpose -> 2x16 (row0,row1)
row0: b0 b1 ... b15         row1: c0 c1 ... c15
  pack row0/row1 into 2x64b lanes -> zip1/zip2 -> nxv16i8

smmla consumes 2x8 (A) x 8x2 (B) per 128b segment.
```

## Pseudo-IR (conceptual)

```
// B prepack (per n tile, per k-step)
b_tile   = vector.extract_strided_slice B[k : k+K, n : n+2]
b_t      = vector.transpose b_tile
b_row0   = vector.extract b_t[0, :]
b_row1   = vector.extract b_t[1, :]
b_i64_0  = vector.bitcast b_row0 : vector<16xi8> -> vector<2xi64>
b_i64_1  = vector.bitcast b_row1 : vector<16xi8> -> vector<2xi64>
b_zip    = llvm.aarch64.sve.zip1/zip2 b_i64_0, b_i64_1

// A pack (per m tile, per k-step)
a_row0   = ...
a_row1   = ...
a_zip    = zip1/zip2(bitcast(a_row0), bitcast(a_row1))

acc      = llvm.aarch64.sve.smmla(acc, a_zip, b_zip)
```

## Benchmarking

### One-shot runs

```
OMP_NUM_THREADS=8 \
python python/bench/bench_i8mm.py --compare --grid \
  --m-list 128,256 --n-list 1024,2048,4096 --k-list 1024,2048,4096 \
  --block-m-list 32 --block-n-list 32 --block-k-list 16 \
  --iters 50 --warmup 10 \
  --save-summary /tmp/i8mm/i8mm_throughput.json
```

### Full sweep script

```
bash python/bench/run_i8mm_all.sh
```

This script runs three scenarios (throughput, small, sweep) and saves
timestamped summary JSON files under `/tmp/i8mm/`.

### Summary output

Each block prints:

- enabled vs disabled summary (same run)
- optional `vs last` (previous summary)
- optional `vs last enabled` (enabled vs previous enabled-only)

The `--save-summary` output is timestamped automatically. If you omit
`--compare-summary`, the script will auto-compare against the most recent
summary with the same base name.

## Correctness validation

The script `python/bench/verify_i8mm.py` runs correctness checks against
`int32` reference results.

The unit test `python/test/unit/cpu/test_i8mm_dot.py` is a lightweight
regression check (fast, small shapes) and should remain enabled to catch
basic correctness issues early.

Run it with:

```
pytest -q python/test/unit/cpu/test_i8mm_dot.py
```

Example:

```
python python/bench/verify_i8mm.py \
  --sizes 64x64x16,128x128x32 \
  --block-m-list 16,32 --block-n-list 16,32 --block-k-list 16 \
  --trials 3 --random-cases 5 \
  --min-m 64 --max-m 256 --min-n 64 --max-n 256 --min-k 16 --max-k 256 \
  --noncontig
```

### Strict presets

For stronger coverage (non-divisible sizes + noncontig + patterns), use
the strict presets:

```
python python/bench/verify_i8mm.py --preset strict \
  --block-m-list 16,32 --block-n-list 16,32 --block-k-list 16
```

Additional strict variants:

```
python python/bench/verify_i8mm.py --preset strict_hugeK \
  --block-m-list 16,32 --block-n-list 16,32 --block-k-list 16

python python/bench/verify_i8mm.py --preset strict_smallM \
  --block-m-list 16,32 --block-n-list 16,32 --block-k-list 16

python python/bench/verify_i8mm.py --preset strict_oddK \
  --block-m-list 16,32 --block-n-list 16,32 --block-k-list 16
```

## Notes / observations

- The best gains tend to appear for larger blocks and larger M/N/K sizes.
- Small blocks or small M may show limited gains.
- Numbers are system-dependent; always benchmark on target hardware.

## Files touched

- `third_party/cpu/lib/TritonCPUTransforms/ConvertDotOp/ConvertDotToSVE2I8MM.cpp`
- `python/bench/bench_i8mm.py`
- `python/bench/run_i8mm_all.sh`
- `python/bench/verify_i8mm.py`
