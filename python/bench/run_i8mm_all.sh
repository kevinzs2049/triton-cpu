#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
LOG_DIR="/tmp/i8mm"
mkdir -p "${LOG_DIR}"

python python/bench/bench_i8mm.py --compare --grid \
  --m-list 128,256 --n-list 1024,2048,4096 --k-list 1024,2048,4096 \
  --block-m-list 32 --block-n-list 32 --block-k-list 16 \
  --iters 30 --warmup 10 \
  --save-summary "${LOG_DIR}/i8mm_throughput.json"

python python/bench/bench_i8mm.py --compare --grid \
  --m-list 64,128 --n-list 1024,2048 --k-list 1024,2048 \
  --block-m-list 16 --block-n-list 16 --block-k-list 16 \
  --iters 30 --warmup 10 \
  --save-summary "${LOG_DIR}/i8mm_small.json"

python python/bench/bench_i8mm.py --compare --grid \
  --m-list 128,256 --n-list 1024,2048 --k-list 1024,2048 \
  --block-m-list 16,32 --block-n-list 16,32 --block-k-list 16 \
  --iters 30 --warmup 10 \
  --save-summary "${LOG_DIR}/i8mm_sweep.json"
