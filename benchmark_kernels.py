"""
Tier 1: equal-effort GEMM microbenchmarks.

Compares:
  FP32 BLAS          — optimal full-precision path
  INT32 MULT         — naive integer multiply baseline (CPU ext)
  Ternary MUX        — BitNet-style add/negate
  Quaternary SHIFT   — power-of-two shift path
  Uniform 2-bit LUT  — non-shift 2-bit (uses MUL)

Usage:
  python benchmark_kernels.py
  python benchmark_kernels.py --device cuda
"""

from __future__ import annotations

import argparse
import time

import torch

from specialized import (
    pack_ternary_int8,
    pack_quaternary_2bit,
    pack_uniform_2bit,
    ternary_matmul,
    shift_matmul,
    uniform_matmul,
    int32_matmul,
    backend_info,
)


def make_random_quaternary_weight(K: int, N: int, c: float) -> torch.Tensor:
    w = torch.randn(K, N)
    boundary = (1.0 + c) / 2.0
    out = torch.empty_like(w)
    out[w > boundary] = 1.0
    out[(w > 0) & (w <= boundary)] = c
    out[(w > -boundary) & (w <= 0)] = -c
    out[w <= -boundary] = -1.0
    return out


def make_random_ternary_weight(K: int, N: int) -> torch.Tensor:
    return torch.randint(-1, 2, (K, N), dtype=torch.float32)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters * 1000


def benchmark_shape(M, K, N, device, warmup=5, iters=30):
    print(f"\n{'='*64}")
    print(f"  M={M}  K={K}  N={N}  device={device}")
    print(f"{'='*64}")

    a_fp = torch.randn(M, K, device=device, dtype=torch.float32)
    w_fp = torch.randn(K, N, device=device, dtype=torch.float32)
    a_int = (a_fp * 64).round().clamp(-127, 127).to(torch.int32)

    c_val = 0.5
    w_quat = make_random_quaternary_weight(K, N, c_val).to(device)
    w_tern = make_random_ternary_weight(K, N).to(device)
    w_uni = w_quat.clone()  # reuse shape; levels remapped by pack

    packed_q = pack_quaternary_2bit(w_quat.cpu().float(), c_val).to(device)
    codes_t = pack_ternary_int8(w_tern.cpu()).to(device)
    packed_u = pack_uniform_2bit(w_uni.cpu().float()).to(device)

    macs = 2.0 * M * K * N

    results = {}

    # FP32 BLAS
    t = _time(lambda: a_fp @ w_fp, warmup, iters, device)
    results["fp32_blas"] = t
    print(f"  {'FP32 BLAS':<22} {t:8.3f} ms  {macs/(t/1000)/1e9:8.2f} GMAC/s")

    # INT32 MULT (CPU only meaningful with ext)
    if device.type == "cpu":
        w_int = (w_fp * 64).round().clamp(-127, 127).to(torch.int32)
        try:
            t = _time(lambda: int32_matmul(a_int, w_int), warmup, iters, device)
            results["int32_mult"] = t
            print(f"  {'INT32 MULT':<22} {t:8.3f} ms  {macs/(t/1000)/1e9:8.2f} GMAC/s  "
                  f"(vs FP32 {results['fp32_blas']/t:.2f}x)")
        except Exception as e:
            print(f"  INT32 MULT skipped: {e}")

    # Ternary
    t = _time(lambda: ternary_matmul(a_int, codes_t), warmup, iters, device)
    results["ternary"] = t
    print(f"  {'Ternary MUX':<22} {t:8.3f} ms  {macs/(t/1000)/1e9:8.2f} GMAC/s  "
          f"(vs FP32 {results['fp32_blas']/t:.2f}x)")

    # Shift
    t = _time(lambda: shift_matmul(a_int, packed_q, 1, K), warmup, iters, device)
    results["shift"] = t
    print(f"  {'Quaternary SHIFT':<22} {t:8.3f} ms  {macs/(t/1000)/1e9:8.2f} GMAC/s  "
          f"(vs FP32 {results['fp32_blas']/t:.2f}x)")

    # Uniform
    t = _time(lambda: uniform_matmul(a_int, packed_u, K), warmup, iters, device)
    results["uniform"] = t
    print(f"  {'Uniform 2-bit LUT':<22} {t:8.3f} ms  {macs/(t/1000)/1e9:8.2f} GMAC/s  "
          f"(vs FP32 {results['fp32_blas']/t:.2f}x)")

    if "int32_mult" in results:
        base = results["int32_mult"]
        print(f"\n  vs INT32 MULT:  ternary {base/results['ternary']:.2f}x  "
              f"shift {base/results['shift']:.2f}x  uniform {base/results['uniform']:.2f}x")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    print("=" * 64)
    print("  TIER 1 — Specialized GEMM microbenchmarks")
    print("  backends:", backend_info())
    print("=" * 64)

    shapes = [
        (1, 256, 256),
        (1, 256, 768),
        (8, 256, 256),
        (1, 768, 768),
        (1, 768, 2048),
        (32, 256, 256),
    ]

    for M, K, N in shapes:
        try:
            benchmark_shape(M, K, N, device, args.warmup, args.iters)
        except Exception as e:
            print(f"  FAILED ({M},{K},{N}): {e}")
            import traceback
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
