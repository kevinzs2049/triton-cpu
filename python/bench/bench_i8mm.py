#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import tempfile
from pathlib import Path

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
    tl.multiple_of(offs_m, BLOCK_M)
    tl.multiple_of(offs_n, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        tl.multiple_of(offs_k, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am +
                    offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk +
                    offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc)


def _dump_root():
    dump_dir = os.getenv("TRITON_DUMP_DIR", "").strip()
    if dump_dir:
        return Path(dump_dir)
    home = Path(os.getenv("TRITON_HOME", Path.home()))
    return home / ".triton" / "dump"


def _cache_root():
    cache_dir = os.getenv("TRITON_CACHE_DIR", "").strip()
    if cache_dir:
        return Path(cache_dir)
    home = Path(os.getenv("TRITON_HOME", Path.home()))
    return home / ".triton" / "cache"


def _scan_asm_for_smmla():
    candidates = []
    dump_root = _dump_root()
    if not dump_root.exists():
        return False, None
    now = time.time()
    for path in dump_root.rglob("*.asm"):
        try:
            if now - path.stat().st_mtime > 600:
                continue
            candidates.append(path)
        except OSError:
            continue
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates[:50]:
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if "smmla" in text or "mmla" in text:
            return True, str(path)
    return False, (str(candidates[0]) if candidates else None)


def _clear_triton_cache():
    shutil.rmtree(_cache_root(), ignore_errors=True)
    shutil.rmtree(_dump_root(), ignore_errors=True)


def run_case(M, N, K, block_m, block_n, block_k, warmup, iters):
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8)
    b = torch.randint(-128, 127, (K, N), dtype=torch.int8)
    c = torch.empty((M, N), dtype=torch.int32)

    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    for _ in range(warmup):
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

    start = time.perf_counter()
    for _ in range(iters):
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
    end = time.perf_counter()

    avg_ms = (end - start) * 1e3 / iters
    gops = (2.0 * M * N * K) / (avg_ms * 1e6)
    return avg_ms, gops


def _parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]

def _read_csv(path):
    rows = {}
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        if header[:6] != ["BM", "BN", "BK", "M", "N", "K"]:
            raise ValueError(f"unexpected CSV header in {path}: {header}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            bm, bn, bk, m, n, k, avg_ms, gops = line.split(",")
            rows[(int(m), int(n), int(k))] = (float(avg_ms), float(gops))
    return rows


def _summarize_compare(enabled_rows, disabled_rows):
    keys = sorted(set(enabled_rows) & set(disabled_rows))
    if not keys:
        print("summary: no common results to compare")
        return None
    speedups = []
    for key in keys:
        _, g_en = enabled_rows[key]
        _, g_off = disabled_rows[key]
        if g_off == 0:
            continue
        speedups.append((key, (g_en / g_off - 1.0) * 100.0))
    if not speedups:
        print("summary: no valid speedup values")
        return None
    speedups.sort(key=lambda x: x[1])
    vals = [v for _, v in speedups]
    avg = sum(vals) / len(vals)
    median = vals[len(vals) // 2]
    improved = sum(1 for v in vals if v > 0)
    worst_key, worst = speedups[0]
    best_key, best = speedups[-1]
    print(
        f"summary: {improved}/{len(vals)} improved | avg {avg:+.2f}% | median {median:+.2f}%"
    )
    print(
        f"best: {best_key[0]}x{best_key[1]}x{best_key[2]} {best:+.2f}% | "
        f"worst: {worst_key[0]}x{worst_key[1]}x{worst_key[2]} {worst:+.2f}%"
    )
    return {
        "avg": avg,
        "median": median,
        "improved": improved,
        "total": len(vals),
        "best": {"shape": list(best_key), "pct": best},
        "worst": {"shape": list(worst_key), "pct": worst},
    }


def _compare_summary(prev, curr):
    if not prev or not curr:
        print("vs last: no previous summary")
        return
    avg_delta = curr["avg"] - prev["avg"]
    median_delta = curr["median"] - prev["median"]
    improved_delta = curr["improved"] - prev["improved"]
    print(
        f"vs last: avg {avg_delta:+.2f}% | median {median_delta:+.2f}% | "
        f"improved {improved_delta:+d} cases"
    )


def _summarize_enabled(rows):
    keys = sorted(rows.keys())
    if not keys:
        return None
    gops_vals = [rows[k][1] for k in keys]
    gops_vals.sort()
    avg = sum(gops_vals) / len(gops_vals)
    median = gops_vals[len(gops_vals) // 2]
    best_key = max(keys, key=lambda k: rows[k][1])
    worst_key = min(keys, key=lambda k: rows[k][1])
    return {
        "avg_gops": avg,
        "median_gops": median,
        "best": {"shape": list(best_key), "gops": rows[best_key][1]},
        "worst": {"shape": list(worst_key), "gops": rows[worst_key][1]},
        "total": len(keys),
    }


def _compare_enabled(prev_enabled, curr_enabled):
    if not prev_enabled or not curr_enabled:
        print("vs last enabled: no previous enabled summary")
        return
    if prev_enabled.get("avg_gops", 0) == 0 or prev_enabled.get("median_gops", 0) == 0:
        return
    avg_delta = (curr_enabled["avg_gops"] / prev_enabled["avg_gops"] - 1.0) * 100.0
    median_delta = (curr_enabled["median_gops"] / prev_enabled["median_gops"] - 1.0) * 100.0
    print(f"vs last enabled: avg {avg_delta:+.2f}% | median {median_delta:+.2f}%")

def _find_latest_summary(base_path):
    base = Path(base_path)
    if not base.parent.exists():
        return None
    stem = base.stem
    matches = sorted(
        base.parent.glob(f"{stem}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return str(matches[0]) if matches else None


def _preset_sizes(name):
    if name == "llm_small":
        m_list = [1, 2, 4, 8, 16, 32, 64]
        nk_list = [1024, 2048, 4096]
        return [(m, n, k) for m in m_list for n in nk_list for k in nk_list]
    if name == "llm_throughput":
        m_list = [128, 256, 512]
        nk_list = [1024, 2048, 4096]
        return [(m, n, k) for m in m_list for n in nk_list for k in nk_list]
    if name == "llm_mixed":
        m_list = [1, 4, 16, 64, 128, 256]
        nk_list = [1024, 2048, 4096]
        return [(m, n, k) for m in m_list for n in nk_list for k in nk_list]
    raise ValueError(f"unknown preset: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="", help="Comma-separated MxNxK list, e.g. 1024x1024x1024,512x1024x2048")
    parser.add_argument("--grid", action="store_true", help="Use grid with --m-list/--n-list/--k-list")
    parser.add_argument("--m-list", default="64,128,256", help="Comma-separated M list for --grid")
    parser.add_argument("--n-list", default="1024,2048,4096", help="Comma-separated N list for --grid")
    parser.add_argument("--k-list", default="1024,2048,4096", help="Comma-separated K list for --grid")
    parser.add_argument("--preset", default="", help="Preset size set: llm_small, llm_throughput, llm_mixed")
    parser.add_argument("--csv", default="", help="Write per-case results to CSV")
    parser.add_argument("--block-m", type=int, default=4, help="Block M for kernel")
    parser.add_argument("--block-n", type=int, default=4, help="Block N for kernel")
    parser.add_argument("--block-k", type=int, default=8, help="Block K for kernel")
    parser.add_argument("--block-m-list", default="", help="Comma-separated block M list")
    parser.add_argument("--block-n-list", default="", help="Comma-separated block N list")
    parser.add_argument("--block-k-list", default="", help="Comma-separated block K list")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations")
    parser.add_argument("--compare", action="store_true", help="Run enabled/disabled i8mm comparison")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dump-asm", action="store_true", help="Enable TRITON_KERNEL_DUMP and asm scan")
    parser.add_argument("--save-summary", default="", help="Save summary JSON (auto adds timestamp)")
    parser.add_argument("--compare-summary", default="", help="Compare against a previous summary JSON")
    parser.add_argument("--no-auto-compare", action="store_true", help="Disable auto-compare when save-summary is set")
    args = parser.parse_args()

    # _clear_triton_cache()
    triton.runtime.driver.set_active_to_cpu()

    if args.dump_asm:
        os.environ["TRITON_KERNEL_DUMP"] = "1"

    if args.compare and not args._child:
        prev_summary = None
        compare_path = args.compare_summary
        if not compare_path and args.save_summary and not args.no_auto_compare:
            compare_path = _find_latest_summary(args.save_summary)
        if compare_path:
            try:
                with open(compare_path, "r") as f:
                    prev_summary = json.load(f)
                print(f"compare-summary: {compare_path}")
            except OSError:
                prev_summary = None
        if not args.sizes and not args.preset and not args.grid:
            args.grid = True
        if args.preset:
            sizes = _preset_sizes(args.preset)
            args.sizes = ",".join(f"{m}x{n}x{k}" for (m, n, k) in sizes)
        if args.grid:
            m_list = _parse_int_list(args.m_list)
            n_list = _parse_int_list(args.n_list)
            k_list = _parse_int_list(args.k_list)
            if not (m_list and n_list and k_list):
                raise ValueError("--grid requires --m-list, --n-list, --k-list")
            args.sizes = ",".join(
                f"{m}x{n}x{k}" for m in m_list for n in n_list for k in k_list
            )
        block_m_list = _parse_int_list(args.block_m_list) or [args.block_m]
        block_n_list = _parse_int_list(args.block_n_list) or [args.block_n]
        block_k_list = _parse_int_list(args.block_k_list) or [args.block_k]
        block_cfgs = [
            (bm, bn, bk)
            for bm in block_m_list
            for bn in block_n_list
            for bk in block_k_list
        ]

        summary_out = {}
        for bm, bn, bk in block_cfgs:
            base_args = [
                sys.executable,
                os.path.abspath(__file__),
                "--sizes",
                args.sizes,
                "--block-m",
                str(bm),
                "--block-n",
                str(bn),
                "--block-k",
                str(bk),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--_child",
            ]
            if args.grid:
                base_args.append("--grid")
                base_args.extend(
                    [
                        "--m-list",
                        args.m_list,
                        "--n-list",
                        args.n_list,
                        "--k-list",
                        args.k_list,
                    ]
                )
            if args.dump_asm:
                base_args.append("--dump-asm")
            if args.preset:
                base_args.extend(["--preset", args.preset])
            if args.csv:
                csv_enabled = f"{args.csv}.enabled"
                csv_disabled = f"{args.csv}.disabled"
            else:
                csv_enabled = tempfile.NamedTemporaryFile(
                    prefix="i8mm_enabled_", suffix=".csv", delete=False
                ).name
                csv_disabled = tempfile.NamedTemporaryFile(
                    prefix="i8mm_disabled_", suffix=".csv", delete=False
                ).name
            base_args.extend(["--csv", csv_enabled])
            env_on = os.environ.copy()
            env_off = os.environ.copy()
            env_on["TRITON_ALWAYS_COMPILE"] = "1"
            env_off["TRITON_ALWAYS_COMPILE"] = "1"
            env_off["TRITON_CPU_DISABLE_SVE2_I8MM"] = "1"

            print(f"=== block {bm}x{bn}x{bk} | i8mm enabled ===")
            subprocess.run(base_args, env=env_on, check=True)
            print(f"=== block {bm}x{bn}x{bk} | i8mm disabled ===")
            base_args_off = base_args[:-2] + ["--csv", csv_disabled]
            subprocess.run(base_args_off, env=env_off, check=True)
            enabled_rows = _read_csv(csv_enabled)
            disabled_rows = _read_csv(csv_disabled)
            curr_summary = _summarize_compare(enabled_rows, disabled_rows)
            enabled_summary = _summarize_enabled(enabled_rows)
            if curr_summary is not None:
                key = f"{bm}x{bn}x{bk}"
                summary_out[key] = curr_summary
                if enabled_summary is not None:
                    summary_out[key]["enabled"] = enabled_summary
                if prev_summary and key in prev_summary:
                    _compare_summary(prev_summary.get(key), curr_summary)
                    if "enabled" in prev_summary[key] and enabled_summary is not None:
                        _compare_enabled(prev_summary[key].get("enabled"), enabled_summary)
        if args.save_summary:
            base, ext = os.path.splitext(args.save_summary)
            if not ext:
                ext = ".json"
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = f"{base}_{ts}{ext}"
            with open(out_path, "w") as f:
                json.dump(summary_out, f, indent=2, sort_keys=True)
            print(f"saved summary: {out_path}")
        return

    print("TRITON_CPU_DISABLE_SVE2_I8MM:", os.getenv("TRITON_CPU_DISABLE_SVE2_I8MM"))
    print("OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"))
    if os.getenv("OMP_PROC_BIND") or os.getenv("OMP_PLACES"):
        print(
            "warning: OMP_PROC_BIND/OMP_PLACES set; pinning can drastically hurt perf on some systems"
        )

    sizes = []
    if not args.sizes and not args.preset and not args.grid:
        args.grid = True
    if args.preset:
        sizes = _preset_sizes(args.preset)
    elif args.grid:
        m_list = _parse_int_list(args.m_list)
        n_list = _parse_int_list(args.n_list)
        k_list = _parse_int_list(args.k_list)
        if not (m_list and n_list and k_list):
            raise ValueError("--grid requires --m-list, --n-list, --k-list")
        sizes = [(m, n, k) for m in m_list for n in n_list for k in k_list]
    else:
        for spec in args.sizes.split(","):
            m_str, n_str, k_str = spec.split("x")
            sizes.append((int(m_str), int(n_str), int(k_str)))

    block_m_list = _parse_int_list(args.block_m_list) or [args.block_m]
    block_n_list = _parse_int_list(args.block_n_list) or [args.block_n]
    block_k_list = _parse_int_list(args.block_k_list) or [args.block_k]
    block_cfgs = [
        (bm, bn, bk)
        for bm in block_m_list
        for bn in block_n_list
        for bk in block_k_list
    ]

    csv_lines = []
    total = len(sizes)
    for bm, bn, bk in block_cfgs:
        print(f"=== block {bm}x{bn}x{bk} ===")
        for idx, (m, n, k) in enumerate(sizes, start=1):
            if (m % bm) or (n % bn) or (k % bk):
                print(
                    f"skip {m}x{n}x{k} (requires M%{bm}, N%{bn}, K%{bk} == 0)"
                )
                continue
            print(f"[{idx}/{total}] running {m}x{n}x{k} ...", flush=True)
            avg_ms, gops = run_case(
                m,
                n,
                k,
                bm,
                bn,
                bk,
                args.warmup,
                args.iters,
            )
            print(f"{m}x{n}x{k}: avg {avg_ms:.3f} ms  |  {gops:.2f} GOPS")
            if args.csv:
                csv_lines.append(
                    f"{bm},{bn},{bk},{m},{n},{k},{avg_ms:.6f},{gops:.4f}"
                )
            if args.dump_asm:
                found, path = _scan_asm_for_smmla()
                print("asm smmla:", "yes" if found else "no", "| file:", path)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("BM,BN,BK,M,N,K,avg_ms,gops\n")
            f.write("\n".join(csv_lines))


if __name__ == "__main__":
    main()
