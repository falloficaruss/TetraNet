"""
Tier 2: end-to-end specialized inference benchmarks on all checkpoints.

Each baseline uses its optimal backend:
  full_precision → FP32 BLAS
  bitnet         → ternary MUX
  fixed_c_* / learned_c → quaternary SHIFT
  uniform_2bit   → 2-bit LUT

Usage:
  python run_benchmarks.py
  python run_benchmarks.py --device cuda --max-new-tokens 64 --n-runs 3
"""

from __future__ import annotations

import argparse
import time

import torch

from inference_engine import load_and_optimize, benchmark_generation, generate
from specialized import backend_info

ALL_BASELINES = [
    "full_precision",
    "bitnet",
    "uniform_2bit",
    "fixed_c_025",
    "fixed_c_05",
    "learned_c",
]

PROMPT = "Once upon a time there was a little girl named Lily"


def tokenize_prompt(prompt, tokenizer_path="tetranet_tokenizer.json"):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    return torch.tensor(tok.encode(prompt).ids).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Specialized e2e inference benchmarks")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--tokenizer", default="tetranet_tokenizer.json")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    if args.device == "cpu":
        torch.set_num_threads(args.threads)

    print("Kernel backends:", backend_info())

    try:
        prompt_ids = tokenize_prompt(PROMPT, args.tokenizer)
        print(f"Prompt: {PROMPT!r} ({prompt_ids.shape[1]} tokens)")
    except Exception as e:
        print(f"Tokenizer unavailable ({e}), random prompt")
        prompt_ids = torch.randint(0, 100, (1, 16))

    prompt_ids = prompt_ids.to(device)
    results = {}
    param_counts = {}

    for baseline in ALL_BASELINES:
        ckpt_path = f"./checkpoint_{baseline}_final.pt"
        print(f"\n{'='*70}")
        print(f"  {baseline}")
        print(f"{'='*70}")

        try:
            t0 = time.time()
            model = load_and_optimize(
                checkpoint_path=ckpt_path,
                baseline=baseline,
                device=str(device),
                enable_kv=True,
            )
            load_s = time.time() - t0
        except FileNotFoundError:
            print(f"  SKIP — missing {ckpt_path}")
            continue
        except Exception as e:
            print(f"  SKIP — load error: {e}")
            import traceback
            traceback.print_exc()
            continue

        n_params = (
            model.get_num_params()
            if hasattr(model, "get_num_params")
            else sum(p.numel() for p in model.parameters())
        )
        # count buffers too for specialized
        n_buf = sum(b.numel() for b in model.buffers())
        param_counts[baseline] = n_params
        print(f"  params≈{n_params:,}  buffers={n_buf:,}  load={load_s:.1f}s")

        try:
            _ = generate(model, prompt_ids, max_new_tokens=8, temperature=0.0, device=device)
            print("  warmup generate: OK")
        except Exception as e:
            print(f"  warmup FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

        try:
            stats = benchmark_generation(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                n_warmup=args.n_warmup,
                n_runs=args.n_runs,
                temperature=0.0,
                device=device,
            )
            results[baseline] = stats
            print(f"  backend:      {stats['backend']}")
            print(f"  prefill:      {stats['prefill_tok_s']:>10,.1f} tok/s  ({stats['prefill_ms']:.1f} ms)")
            print(f"  decode:       {stats['decode_tok_s']:>10,.1f} tok/s  ({stats['decode_ms']:.1f} ms)")
            print(f"  total:        {stats['total_tok_s']:>10,.1f} tok/s")
            print(f"  peak mem:     {stats['peak_mb']:>10,.1f} MB")
            print(f"  compressed:   {stats['compressed_mb']:>10.2f} MB")
            print(f"  KV-cache:     {stats['has_kv']}")
        except Exception as e:
            print(f"  BENCHMARK FAILED: {e}")
            import traceback
            traceback.print_exc()

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n\n{'='*90}")
    print(f"{'SUMMARY: Specialized Inference (Tier 2)':^90s}")
    print(f"{'='*90}")
    print(f"Device: {device} | prompt={prompt_ids.shape[1]} | gen={args.max_new_tokens}")
    print()
    hdr = (
        f"  {'Baseline':<18s} {'Backend':<20s} {'Prefill':>10s} {'Decode':>10s} "
        f"{'PeakMB':>8s} {'PackMB':>8s}"
    )
    print(hdr)
    print("  " + "-" * 80)
    for b in ALL_BASELINES:
        if b not in results:
            continue
        s = results[b]
        print(
            f"  {b:<18s} {s['backend']:<20s} {s['prefill_tok_s']:>10,.1f} "
            f"{s['decode_tok_s']:>10,.1f} {s['peak_mb']:>8.1f} {s['compressed_mb']:>8.2f}"
        )
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
