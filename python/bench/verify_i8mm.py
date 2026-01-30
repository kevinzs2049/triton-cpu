#!/usr/bin/env python3
import argparse
import os
import random
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def i8mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am +
                    offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk +
                    offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc)


def _parse_sizes(text):
    sizes = []
    for spec in text.split(","):
        spec = spec.strip()
        if not spec:
            continue
        m_str, n_str, k_str = spec.split("x")
        sizes.append((int(m_str), int(n_str), int(k_str)))
    return sizes


def _parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def _run_case(M, N, K, block_m, block_n, block_k, seed, noncontig):
    torch.manual_seed(seed)
    if noncontig:
        a_base = torch.randint(-128, 127, (K, M), dtype=torch.int8)
        b_base = torch.randint(-128, 127, (N, K), dtype=torch.int8)
        a = a_base.t()
        b = b_base.t()
    else:
        a = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        b = torch.randint(-128, 127, (K, N), dtype=torch.int8)
    c = torch.empty((M, N), dtype=torch.int32)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    i8mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )

    ref = a.to(torch.int32) @ b.to(torch.int32)
    diff = (c - ref).abs()
    max_diff = int(diff.max().item())
    return max_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="64x64x16,64x64x32,128x128x16,128x128x32,256x128x16",
        help="Comma-separated MxNxK list",
    )
    parser.add_argument("--block-m-list", default="16,32", help="Block M list")
    parser.add_argument("--block-n-list", default="16,32", help="Block N list")
    parser.add_argument("--block-k-list", default="16", help="Block K list")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--trials", type=int, default=1, help="Trials per case")
    parser.add_argument("--random-cases", type=int, default=0, help="Random case count")
    parser.add_argument("--min-m", type=int, default=64)
    parser.add_argument("--max-m", type=int, default=256)
    parser.add_argument("--min-n", type=int, default=64)
    parser.add_argument("--max-n", type=int, default=256)
    parser.add_argument("--min-k", type=int, default=16)
    parser.add_argument("--max-k", type=int, default=256)
    parser.add_argument("--noncontig", action="store_true", help="Use non-contiguous A/B")
    parser.add_argument(
        "--disable-i8mm",
        action="store_true",
        help="Disable SVE2 i8mm pass (TRITON_CPU_DISABLE_SVE2_I8MM=1)",
    )
    args = parser.parse_args()

    if args.disable_i8mm:
        os.environ["TRITON_CPU_DISABLE_SVE2_I8MM"] = "1"
    else:
        os.environ.pop("TRITON_CPU_DISABLE_SVE2_I8MM", None)

    triton.runtime.driver.set_active_to_cpu()

    sizes = _parse_sizes(args.sizes)
    bm_list = _parse_int_list(args.block_m_list) or [16]
    bn_list = _parse_int_list(args.block_n_list) or [16]
    bk_list = _parse_int_list(args.block_k_list) or [16]
    if args.random_cases:
        def lcm(a, b):
            return a * b // (a % b == 0 and b or __import__("math").gcd(a, b))
        m_align = 1
        n_align = 1
        k_align = 1
        for v in bm_list:
            m_align = lcm(m_align, v)
        for v in bn_list:
            n_align = lcm(n_align, v)
        for v in bk_list:
            k_align = lcm(k_align, v)
        m_align = lcm(m_align, 4)
        n_align = lcm(n_align, 4)
        k_align = lcm(k_align, 8)
        rng = random.Random(args.seed)
        for _ in range(args.random_cases):
            m = rng.randrange(args.min_m, args.max_m + 1, m_align)
            n = rng.randrange(args.min_n, args.max_n + 1, n_align)
            k = rng.randrange(args.min_k, args.max_k + 1, k_align)
            sizes.append((m, n, k))

    total = 0
    failed = 0
    for bm in bm_list:
        for bn in bn_list:
            for bk in bk_list:
                print(f"=== block {bm}x{bn}x{bk} ===")
                for (m, n, k) in sizes:
                    if (m % bm) or (n % bn) or (k % bk):
                        print(f"skip {m}x{n}x{k} (divisible by {bm},{bn},{bk})")
                        continue
                    total += 1
                    max_diff = 0
                    for t in range(args.trials):
                        max_diff = max(
                            max_diff,
                            _run_case(
                                m, n, k, bm, bn, bk, args.seed + t, args.noncontig
                            ),
                        )
                    status = "OK" if max_diff == 0 else "FAIL"
                    if max_diff != 0:
                        failed += 1
                    print(f"{m}x{n}x{k}: {status} max_diff={max_diff}")

    print(f"summary: {total - failed}/{total} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
