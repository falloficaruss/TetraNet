"""
Benchmark compute cost of all baselines: MAC counting, energy estimation, CPU/GPU throughput.

Usage:
    python benchmark.py                                    # all baselines, single-thread CPU
    python benchmark.py --baselines bitnet fixed_c_05       # subset
    python benchmark.py --threads 0                         # all CPU cores (peak throughput)
    python benchmark.py --iterations 10                     # quick run (fewer iterations)

Output: clean table comparing full_precision, bitnet, fixed_c_05, fixed_c_025.
"""
import argparse
import os
import time

import torch

from train_kaggle import build_model

DEFAULT_CONFIG = dict(
    vocab_size=4096,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    ffn_dim=3072,
    max_seq_len=512,
    tie_weights=True,
)

ALL_BASELINES = ["full_precision", "bitnet", "fixed_c_05", "fixed_c_025"]

# ── Energy per op (Horowitz ISSCC 2014, 45nm CMOS, pJ) ──
FP32_MULT = 3.7
FP32_ADD = 0.9
INT8_MULT = 0.2
INT8_ADD = 0.03
SHIFT = 0.1  # ~AND gate energy, rough

# Per-MAC energy estimate for each baseline's quantized multiply-add
QUANT_ENERGY = {
    "full_precision": FP32_MULT + FP32_ADD,         # 4.6 pJ — standard FP MAC
    "bitnet": FP32_ADD + 0.1,                       # 1.0 pJ — MUX/negate + ADD, no MULT
    "fixed_c_05": FP32_ADD + SHIFT + 0.1,           # 1.1 pJ — shift-1 + MUX + ADD
    "fixed_c_025": FP32_ADD + SHIFT + SHIFT + 0.1,  # 1.2 pJ — shift-2 + MUX + ADD
}


def count_macs_per_token(cfg: dict) -> dict:
    """Compute MACs per token for each component of the model."""
    h = cfg["hidden_dim"]
    ffn = cfg["ffn_dim"]
    T = cfg["max_seq_len"]
    n_heads = cfg["num_heads"]
    d_head = h // n_heads
    n_layers = cfg["num_layers"]
    vocab = cfg["vocab_size"]

    # Attention per layer
    qkv_proj = 3 * h * h                        # Q, K, V projections
    qk_dot = n_heads * T * d_head               # Q @ K^T, per token
    attn_v = n_heads * T * d_head               # attn @ V, per token
    o_proj = h * h                               # output projection
    attn_total = qkv_proj + qk_dot + attn_v + o_proj

    # FFN per layer (SwiGLU: gate + up + down)
    ffn_total = h * ffn + h * ffn + ffn * h    # gate + up + down

    per_layer = attn_total + ffn_total
    all_layers = per_layer * n_layers

    # Embedding = lookup (0 MACs)
    # Final norm = h (element-wise, negligible)
    # LM head
    lm_head = h * vocab

    total = all_layers + lm_head

    return {
        "attention_per_layer": attn_total,
        "ffn_per_layer": ffn_total,
        "per_layer": per_layer,
        "n_layers": n_layers,
        "all_layers": all_layers,
        "lm_head": lm_head,
        "total_macs": total,
    }


def baseline_quant_type(name: str) -> str:
    types = {
        "full_precision": "FP32 MULT + ADD",
        "bitnet": "MUX / NEG + FP32 ADD",
        "fixed_c_05": "SHIFT-1 + MUX / NEG + FP32 ADD",
        "fixed_c_025": "SHIFT-2 + MUX / NEG + FP32 ADD",
    }
    return types.get(name, "?")


def print_mac_table(macs: dict):
    print("─── MAC count per token ───")
    print(f"  Attention (per layer):     {macs['attention_per_layer']:>12,}")
    print(f"  FFN (per layer):           {macs['ffn_per_layer']:>12,}")
    print(f"  Total per layer:           {macs['per_layer']:>12,}")
    print(f"  × {macs['n_layers']} layers:                {macs['all_layers']:>12,}")
    print(f"  LM head:                   {macs['lm_head']:>12,}")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL MACs/token:          {macs['total_macs']:>12,}")
    print(f"  TOTAL FLOPs/token:         {macs['total_macs'] * 2:>12,}")
    print()


def print_energy_table(macs_total: int, baselines: list[str]):
    print("─── Energy estimate per token (Horowitz ISSCC 2014, 45nm) ───")
    print(f"  {'Baseline':<20s} {'MACs/tok':<12s} {'pJ/MAC':<10s} {'nJ/tok':<10s} {'vs FP':<10s}")
    print(f"  {'─'*20} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    fp_energy = macs_total * QUANT_ENERGY["full_precision"]
    for name in baselines:
        pj_per_mac = QUANT_ENERGY[name]
        nJ = macs_total * pj_per_mac / 1000  # convert pJ to nJ
        ratio = nJ / (fp_energy / 1000) * 100 if name != "full_precision" else 100
        print(f"  {name:<20s} {macs_total:<12,} {pj_per_mac:<10.1f} {nJ:<10.1f} {ratio:<10.1f}%")
    print()


def benchmark_throughput(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    n_warmup: int = 10,
    n_iter: int = 100,
    device: str = "cpu",
):
    """Measure tok/s for a single model on random inputs. Works on CPU and GPU."""
    model = model.to(device)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            model(input_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    elapsed_ms = elapsed * 1000 / n_iter  # ms per forward
    tokens_per_sec = batch_size * seq_len / (elapsed_ms / 1000)
    return tokens_per_sec, elapsed_ms


def print_throughput_table(results: dict, baselines: list[str], batch_sizes: list[int], device: str):
    print(f"─── Throughput on {device} (synthetic random inputs) ───")
    header = f"  {'Baseline':<20s}"
    for bs in batch_sizes:
        header += f" {'tok/s (bs=' + str(bs) + ')':>18s}"
    header += f" {'ms/fwd':>10s}"
    print(header)
    print(f"  {'─'*20} {'─'*18 * len(batch_sizes)} {'─'*10}")
    for name in baselines:
        row = f"  {name:<20s}"
        for bs in batch_sizes:
            tok_s, ms = results[name][bs]
            row += f" {tok_s:>18,.0f}"
        row += f" {results[name][batch_sizes[0]][1]:>10.2f}"
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark compute cost of all baselines"
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=ALL_BASELINES,
        choices=ALL_BASELINES,
        help="Baselines to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 8, 64],
        help="Batch sizes for throughput test",
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="CPU threads (0 = all cores). Single-thread for edge-CPU latency.",
    )
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="Forward passes per benchmark (lower = faster, noisier)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"]
    )
    parser.add_argument(
        "--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"]
    )
    parser.add_argument("--ffn-dim", type=int, default=DEFAULT_CONFIG["ffn_dim"])
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG["max_seq_len"])
    parser.add_argument(
        "--num-heads", type=int, default=DEFAULT_CONFIG["num_heads"]
    )
    args = parser.parse_args()

    cfg = dict(
        vocab_size=DEFAULT_CONFIG["vocab_size"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.seq_len,
        tie_weights=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Thread control for CPU
    n_threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    if device == "cpu":
        torch.set_num_threads(n_threads)
    thread_info = f" ({n_threads} thread{'s' if n_threads > 1 else ''})" if device == "cpu" else ""

    macs = count_macs_per_token(cfg)

    print(f"Model: hidden_dim={cfg['hidden_dim']}, num_layers={cfg['num_layers']}, "
          f"ffn_dim={cfg['ffn_dim']}, seq_len={cfg['max_seq_len']}")
    print(f"Device: {device}{thread_info}")
    print()

    print_mac_table(macs)
    print_energy_table(macs["total_macs"], args.baselines)

    print("─── Building models for throughput benchmark ───")
    throughput_results = {}
    for name in args.baselines:
        print(f"  Building {name}...", flush=True)
        model = build_model(
            baseline=name,
            vocab_size=cfg["vocab_size"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            ffn_dim=cfg["ffn_dim"],
            max_seq_len=cfg["max_seq_len"],
            tie_weights=True,
        )
        throughput_results[name] = {}
        for bs in args.batch_sizes:
            tok_s, ms = benchmark_throughput(
                model, bs, cfg["max_seq_len"], cfg["vocab_size"],
                n_iter=args.iterations,
                device=device,
            )
            throughput_results[name][bs] = (tok_s, ms)
    print()
    print_throughput_table(throughput_results, args.baselines, args.batch_sizes, device)

    # ── Summary table ──
    print("=" * 80)
    print(f"{'SUMMARY':^80s}")
    print("=" * 80)
    header = f"  {'Baseline':<20s} {'MACs/tok':<12s} {'Quant op':<32s} {'pJ/MAC':<10s} {'nJ/tok':<10s} {'tok/s(bs=1)':>14s}"
    print(header)
    print(f"  {'─'*20} {'─'*12} {'─'*32} {'─'*10} {'─'*10} {'─'*14}")
    for name in args.baselines:
        total_macs = macs["total_macs"]
        pj_per_mac = QUANT_ENERGY[name]
        nJ = total_macs * pj_per_mac / 1000
        tok_s, _ = throughput_results[name][1]
        row = (f"  {name:<20s} {total_macs:<12,} {baseline_quant_type(name):<32s}"
               f" {pj_per_mac:<10.1f} {nJ:<10.1f} {tok_s:>14,.0f}")
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
