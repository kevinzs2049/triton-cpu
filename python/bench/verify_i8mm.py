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


@triton.jit
def i8mm_kernel_masked(
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
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=a_mask,
            other=0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=b_mask,
            other=0,
        )
        acc += tl.dot(a, b)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=c_mask,
    )


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


def _gen_tensor(shape, pattern, dtype, seed):
    if pattern is None:
        torch.manual_seed(seed)
        return torch.randint(-128, 127, shape, dtype=dtype)
    if pattern == "zeros":
        return torch.zeros(shape, dtype=dtype)
    if pattern == "ones":
        return torch.ones(shape, dtype=dtype)
    if pattern == "neg_ones":
        return torch.full(shape, -1, dtype=dtype)
    if pattern == "alt":
        data = torch.arange(0, shape[0] * shape[1], dtype=torch.int32)
        data = (data % 2) * 255 - 128
        return data.to(dtype).reshape(shape)
    if pattern == "ramp":
        data = torch.arange(0, shape[0] * shape[1], dtype=torch.int32)
        data = (data % 255) - 128
        return data.to(dtype).reshape(shape)
    raise ValueError(f"unknown pattern: {pattern}")


def _run_case(M, N, K, block_m, block_n, block_k, seed, noncontig, pattern=None, use_mask=False):
    torch.manual_seed(seed)
    if noncontig:
        a_base = _gen_tensor((K, M), pattern, torch.int8, seed)
        b_base = _gen_tensor((N, K), pattern, torch.int8, seed + 1)
        a = a_base.t()
        b = b_base.t()
    else:
        a = _gen_tensor((M, K), pattern, torch.int8, seed)
        b = _gen_tensor((K, N), pattern, torch.int8, seed + 1)
    c = torch.empty((M, N), dtype=torch.int32)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    kernel = i8mm_kernel_masked if use_mask else i8mm_kernel
    kernel[grid](
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
        "--preset",
        default="",
        help="Size preset (strict/strict_hugeK/strict_smallM/strict_oddK)",
    )
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
        "--patterns",
        action="store_true",
        help="Also test deterministic value patterns (zeros/ones/alt/ramp)",
    )
    parser.add_argument(
        "--allow-tail",
        action="store_true",
        help="Allow non-divisible sizes and validate with masked kernel",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Always use masked kernel (even for divisible sizes)",
    )
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

    if args.preset:
        args.trials = max(args.trials, 3)
        args.random_cases = max(args.random_cases, 16)
        args.allow_tail = True
        args.patterns = True
        if args.preset == "strict":
            sizes = [
                (16, 16, 16),
                (32, 32, 16),
                (64, 64, 16),
                (64, 128, 16),
                (128, 64, 16),
                (96, 160, 32),
                (160, 96, 32),
                (128, 128, 32),
                (192, 256, 64),
                (256, 192, 64),
                (256, 256, 128),
                (256, 256, 256),
                (512, 256, 256),
                (256, 512, 256),
                (512, 512, 256),
                (65, 129, 17),
                (127, 191, 33),
                (129, 127, 65),
                (255, 257, 129),
            ]
        elif args.preset == "strict_hugeK":
            sizes = [
                (128, 128, 512),
                (128, 256, 512),
                (256, 128, 512),
                (256, 256, 512),
                (128, 128, 1024),
                (128, 256, 1024),
                (256, 128, 1024),
                (256, 256, 1024),
                (64, 128, 2048),
                (128, 64, 2048),
                (128, 256, 2048),
                (256, 128, 2048),
                (256, 256, 2048),
                (65, 129, 513),
                (127, 191, 1025),
            ]
        elif args.preset == "strict_smallM":
            sizes = [
                (1, 256, 256),
                (2, 256, 256),
                (4, 256, 256),
                (8, 256, 256),
                (16, 256, 256),
                (1, 512, 256),
                (2, 512, 256),
                (4, 512, 256),
                (8, 512, 256),
                (16, 512, 256),
                (1, 256, 512),
                (2, 256, 512),
                (4, 256, 512),
                (8, 256, 512),
                (16, 256, 512),
                (3, 257, 129),
                (5, 511, 257),
            ]
        elif args.preset == "strict_oddK":
            sizes = [
                (64, 64, 17),
                (64, 128, 33),
                (128, 64, 65),
                (128, 128, 127),
                (256, 128, 255),
                (128, 256, 257),
                (256, 256, 257),
                (192, 256, 129),
                (256, 192, 129),
                (65, 129, 17),
                (127, 191, 33),
                (129, 127, 65),
                (255, 257, 129),
            ]
        else:
            raise ValueError(f"unknown preset: {args.preset}")
    else:
        sizes = _parse_sizes(args.sizes)

    if args.preset == "strict":
        layouts = [False, True]
    else:
        layouts = [args.noncontig]
    bm_list = _parse_int_list(args.block_m_list) or [16]
    bn_list = _parse_int_list(args.block_n_list) or [16]
    bk_list = _parse_int_list(args.block_k_list) or [16]
    if args.random_cases:
        rng = random.Random(args.seed)
        if args.allow_tail:
            for _ in range(args.random_cases):
                m = rng.randint(max(4, args.min_m), args.max_m)
                n = rng.randint(max(4, args.min_n), args.max_n)
                k = rng.randint(max(4, args.min_k), args.max_k)
                sizes.append((m, n, k))
        else:
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
                for noncontig in layouts:
                    layout_tag = "noncontig" if noncontig else "contig"
                    if len(layouts) > 1:
                        print(f"-- layout: {layout_tag} --")
                    for (m, n, k) in sizes:
                        if (m % bm) or (n % bn) or (k % bk):
                            if not (args.allow_tail or args.mask):
                                print(f"skip {m}x{n}x{k} (divisible by {bm},{bn},{bk})")
                                continue
                        total += 1
                        max_diff = 0
                        use_mask = args.mask or (args.allow_tail and ((m % bm) or (n % bn) or (k % bk)))
                        for t in range(args.trials):
                            max_diff = max(
                                max_diff,
                                _run_case(
                                    m,
                                    n,
                                    k,
                                    bm,
                                    bn,
                                    bk,
                                    args.seed + t,
                                    noncontig,
                                    None,
                                    use_mask,
                                ),
                            )
                        if args.patterns:
                            for pattern in ("zeros", "ones", "neg_ones", "alt", "ramp"):
                                max_diff = max(
                                    max_diff,
                                    _run_case(
                                        m,
                                        n,
                                        k,
                                        bm,
                                        bn,
                                        bk,
                                        args.seed,
                                        noncontig,
                                        pattern,
                                        use_mask,
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
