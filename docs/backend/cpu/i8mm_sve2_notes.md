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

Example:

```
python python/bench/verify_i8mm.py \
  --sizes 64x64x16,128x128x32 \
  --block-m-list 16,32 --block-n-list 16,32 --block-k-list 16 \
  --trials 3 --random-cases 5 \
  --min-m 64 --max-m 256 --min-n 64 --max-n 256 --min-k 16 --max-k 256 \
  --noncontig
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
