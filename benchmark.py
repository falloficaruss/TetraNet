"""
Compute + memory energy comparison for TetraNet baselines (~10M default).

Analytical model (primary research metric):
  1. MAC split — quantized projections vs always-FP attention vs LM head
  2. Compute energy — Horowitz ISSCC 2014 op energies (45nm CMOS, pJ)
  3. Memory energy — weight bitwidth × DRAM bit energy (decode: re-read weights/token)

Optional measured throughput (secondary, training-style FP forward — not specialized kernels):
  python benchmark.py --throughput

Usage:
  python benchmark.py
  python benchmark.py --csv energy_10m.csv --json energy_10m.json
  python benchmark.py --baselines full_precision bitnet fixed_c_05
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import Optional

import torch

from train_kaggle import build_model

# ── Default: toy / local checkpoint scale (~10M) ──
DEFAULT_CONFIG = dict(
    vocab_size=4096,
    hidden_dim=256,
    num_layers=8,
    num_heads=8,
    ffn_dim=1024,
    max_seq_len=512,
    tie_weights=True,
)

ALL_BASELINES = [
    "full_precision",
    "bitnet",
    "uniform_2bit",
    "fixed_c_025",
    "fixed_c_05",
    "learned_c",
    "heterogeneous",
]

# ═══════════════════════════════════════════════════════════════════════════
#  Op energies — Horowitz ISSCC 2014, 45nm CMOS (pJ)
#  Widely used order-of-magnitude table for ML systems papers.
# ═══════════════════════════════════════════════════════════════════════════
FP32_MULT = 3.7
FP32_ADD = 0.9
INT8_MULT = 0.2
INT8_ADD = 0.03
SHIFT = 0.1          # arithmetic shift ~ cheap integer op
MUX_NEG = 0.1        # conditional select / negate (approx)

# DRAM: 32-bit access ~640 pJ → 20 pJ/bit (Horowitz-style DRAM table)
DRAM_PJ_PER_BIT = 20.0
# On-chip SRAM cache (8KB-class 32b) is much cheaper; report as secondary
SRAM_PJ_PER_BIT = 0.3  # ~10 pJ / 32-bit ≈ 0.3 pJ/bit order

# Per quantized-projection MAC energy by arithmetic path.
# Low-bit paths assume INT accumulate (matches specialized INT8-act inference).
PROJ_MAC_ENERGY_PJ = {
    "full_precision": FP32_MULT + FP32_ADD,                 # 4.6
    "bitnet": MUX_NEG + INT8_ADD,                           # ~0.13  ternary MUX+ADD
    "uniform_2bit": INT8_MULT + INT8_ADD,                   # 0.23   real mul for ±1/3
    "fixed_c_05": SHIFT + MUX_NEG + INT8_ADD,               # ~0.23  >>1 + MUX + ADD
    "fixed_c_025": 2 * SHIFT + MUX_NEG + INT8_ADD,          # ~0.33  >>2 + MUX + ADD
    # learned_c / heterogeneous filled via weighted mix below
}

BASELINE_OP_DESC = {
    "full_precision": "FP32 MULT+ADD",
    "bitnet": "MUX/NEG + ADD (no MULT)",
    "uniform_2bit": "INT MUL + ADD (non-PoT 2-bit)",
    "fixed_c_05": "SHIFT-1 + MUX + ADD",
    "fixed_c_025": "SHIFT-2 + MUX + ADD",
    "learned_c": "SHIFT mix (post-snap 0.25/0.5)",
    "heterogeneous": "SHIFT mix (blueprint)",
}

# Ideal packed bits per weight (plus scales handled separately)
BITS_PER_WEIGHT = {
    "full_precision": 32.0,
    "bitnet": 1.58,          # theoretical ternary entropy; packed storage often 2
    "uniform_2bit": 2.0,
    "fixed_c_05": 2.0,
    "fixed_c_025": 2.0,
    "learned_c": 2.0,
    "heterogeneous": 2.0,
}
BITS_PER_WEIGHT_PACKED = {
    "full_precision": 32.0,
    "bitnet": 2.0,           # commodity 2-bit packing
    "uniform_2bit": 2.0,
    "fixed_c_05": 2.0,
    "fixed_c_025": 2.0,
    "learned_c": 2.0,
    "heterogeneous": 2.0,
}


def count_macs_per_token(cfg: dict, decode: bool = True) -> dict:
    """
    MAC counts per generated token.

    decode=True  — autoregressive step: attn score cost uses cache length ≈ seq context
                   We use T = max_seq_len as worst-case full context (paper upper bound).
    decode=False — prefill amortised per token in a full sequence of length T.
    """
    h = cfg["hidden_dim"]
    ffn = cfg["ffn_dim"]
    T = cfg["max_seq_len"]
    n_heads = cfg["num_heads"]
    d_head = h // n_heads
    n_layers = cfg["num_layers"]
    vocab = cfg["vocab_size"]

    # Quantized linear projections (eligible for specialized arithmetic)
    qkv_proj = 3 * h * h
    o_proj = h * h
    ffn_proj = h * ffn + h * ffn + ffn * h  # gate + up + down
    proj_per_layer = qkv_proj + o_proj + ffn_proj
    quant_proj_macs = proj_per_layer * n_layers

    # Always-FP attention arithmetic (scores + weighted values)
    # Per token with context length T (decode with full cache): O(T * d) per head
    qk_dot = n_heads * T * d_head
    attn_v = n_heads * T * d_head
    fp_attn_per_layer = qk_dot + attn_v
    fp_attn_macs = fp_attn_per_layer * n_layers

    # LM head (FP unless separately quantized; we keep FP)
    lm_head = h * vocab

    # Prefill: same structure; table is per-token so prefill total = T * this for full seq
    total = quant_proj_macs + fp_attn_macs + lm_head

    # Weight parameter counts (for memory energy)
    # nn.Linear weight is [out, in]; count elements
    proj_weight_params = n_layers * (
        4 * h * h  # q,k,v,o
        + 2 * h * ffn  # gate, up
        + ffn * h  # down
    )
    embed_params = vocab * h  # tied with lm_head when tie_weights
    # scales: one FP32 gamma per quantized projection (7 per layer)
    n_quant_modules = n_layers * 7
    scale_params = n_quant_modules  # one scalar each (stored FP32 → 32 bits)

    # Per-projection MAC shares (for heterogeneous / learned mix)
    mac_q = h * h
    mac_k = h * h
    mac_v = h * h
    mac_o = h * h
    mac_gate = h * ffn
    mac_up = h * ffn
    mac_down = ffn * h
    proj_mac_breakdown = {
        "q_proj": mac_q * n_layers,
        "k_proj": mac_k * n_layers,
        "v_proj": mac_v * n_layers,
        "o_proj": mac_o * n_layers,
        "gate_proj": mac_gate * n_layers,
        "up_proj": mac_up * n_layers,
        "down_proj": mac_down * n_layers,
    }
    assert sum(proj_mac_breakdown.values()) == quant_proj_macs

    # Parameter counts per projection type (same layout)
    param_breakdown = {
        "q_proj": mac_q * n_layers,
        "k_proj": mac_k * n_layers,
        "v_proj": mac_v * n_layers,
        "o_proj": mac_o * n_layers,
        "gate_proj": mac_gate * n_layers,
        "up_proj": mac_up * n_layers,
        "down_proj": mac_down * n_layers,
    }

    return {
        "quant_proj_macs": quant_proj_macs,
        "fp_attn_macs": fp_attn_macs,
        "lm_head_macs": lm_head,
        "total_macs": total,
        "proj_per_layer": proj_per_layer,
        "fp_attn_per_layer": fp_attn_per_layer,
        "n_layers": n_layers,
        "proj_weight_params": proj_weight_params,
        "embed_params": embed_params,
        "scale_params": scale_params,
        "n_quant_modules": n_quant_modules,
        "proj_mac_breakdown": proj_mac_breakdown,
        "param_breakdown": param_breakdown,
        "hidden_dim": h,
        "ffn_dim": ffn,
        "seq_len": T,
        "vocab_size": vocab,
    }


# Structural fingerprint / blueprint → c wells
# From findings + HeterogeneousQuaternaryLinear.BLUEPRINT
HETERO_C = {
    "q_proj": 0.5,
    "k_proj": 0.5,
    "v_proj": 0.25,
    "o_proj": 0.25,
    "gate_proj": 0.5,
    "up_proj": 0.25,
    "down_proj": 0.25,
}

# Toy snap fingerprint (FINDINGS.md): Q/K→0.5, V/O→0.25, MLP mixed ~5/8 to 0.5
LEARNED_C_FRAC_05 = {
    "q_proj": 1.0,
    "k_proj": 1.0,
    "v_proj": 1.0 / 8.0,   # 1/8 → 0.5 in findings
    "o_proj": 0.0,
    "gate_proj": 5.0 / 8.0,
    "up_proj": 5.0 / 8.0,
    "down_proj": 5.0 / 8.0,
}


def _shift_mac_energy(c: float) -> float:
    if abs(c - 0.25) <= abs(c - 0.5):
        return PROJ_MAC_ENERGY_PJ["fixed_c_025"]
    return PROJ_MAC_ENERGY_PJ["fixed_c_05"]


def proj_mac_energy_pj(baseline: str, macs: dict) -> float:
    """Average pJ per quantized-projection MAC for this baseline."""
    if baseline in PROJ_MAC_ENERGY_PJ:
        return PROJ_MAC_ENERGY_PJ[baseline]

    bd = macs["proj_mac_breakdown"]
    total = macs["quant_proj_macs"]

    if baseline == "heterogeneous":
        e = 0.0
        for name, n in bd.items():
            e += n * _shift_mac_energy(HETERO_C[name])
        return e / total

    if baseline == "learned_c":
        e = 0.0
        for name, n in bd.items():
            f05 = LEARNED_C_FRAC_05[name]
            e += n * (
                f05 * PROJ_MAC_ENERGY_PJ["fixed_c_05"]
                + (1.0 - f05) * PROJ_MAC_ENERGY_PJ["fixed_c_025"]
            )
        return e / total

    raise KeyError(baseline)


def compute_energy_nj(baseline: str, macs: dict) -> dict:
    """
    Compute energy per token (nJ).

    quant_proj MACs → baseline-specific
    fp_attn + lm_head → FP32 MAC
    """
    e_proj = proj_mac_energy_pj(baseline, macs)
    e_fp = FP32_MULT + FP32_ADD

    pJ_proj = macs["quant_proj_macs"] * e_proj
    pJ_attn = macs["fp_attn_macs"] * e_fp
    pJ_lm = macs["lm_head_macs"] * e_fp
    pJ_total = pJ_proj + pJ_attn + pJ_lm

    return {
        "pj_per_proj_mac": e_proj,
        "compute_proj_nJ": pJ_proj / 1000.0,
        "compute_fp_attn_nJ": pJ_attn / 1000.0,
        "compute_lm_nJ": pJ_lm / 1000.0,
        "compute_total_nJ": pJ_total / 1000.0,
    }


def memory_energy_nj(
    baseline: str,
    macs: dict,
    use_packed_bitnet: bool = False,
    pj_per_bit: float = DRAM_PJ_PER_BIT,
    include_embed: bool = True,
) -> dict:
    """
    Weight memory energy per token (decode: assume all weights re-read once).

    bits = ideal 1.58 for BitNet unless use_packed_bitnet (2-bit commodity packing).
    Scales: n_quant_modules × 32-bit FP32.
    Embeddings: always FP32 (not specialized in our stack).
    """
    bits_table = BITS_PER_WEIGHT_PACKED if use_packed_bitnet else BITS_PER_WEIGHT
    bits_w = bits_table[baseline]

    # Projection weights
    if baseline == "full_precision":
        proj_bits = macs["proj_weight_params"] * 32.0
        scale_bits = 0.0
    else:
        proj_bits = macs["proj_weight_params"] * bits_w
        scale_bits = macs["scale_params"] * 32.0

    embed_bits = macs["embed_params"] * 32.0 if include_embed else 0.0
    total_bits = proj_bits + scale_bits + embed_bits

    return {
        "bits_per_weight": bits_w,
        "proj_weight_Mbit": proj_bits / 1e6,
        "scale_Mbit": scale_bits / 1e6,
        "embed_Mbit": embed_bits / 1e6,
        "total_Mbit": total_bits / 1e6,
        "memory_proj_nJ": proj_bits * pj_per_bit / 1000.0,
        "memory_scale_nJ": scale_bits * pj_per_bit / 1000.0,
        "memory_embed_nJ": embed_bits * pj_per_bit / 1000.0,
        "memory_total_nJ": total_bits * pj_per_bit / 1000.0,
        "pj_per_bit": pj_per_bit,
    }


def estimate_row(baseline: str, macs: dict, packed_bitnet: bool = False) -> dict:
    comp = compute_energy_nj(baseline, macs)
    mem = memory_energy_nj(baseline, macs, use_packed_bitnet=packed_bitnet)
    total = comp["compute_total_nJ"] + mem["memory_total_nJ"]
    return {
        "baseline": baseline,
        "op": BASELINE_OP_DESC[baseline],
        **comp,
        **mem,
        "total_nJ": total,
    }


def print_mac_split(macs: dict):
    print("─── MAC count per token (decode, context=T) ───")
    print(f"  Quantized projections:     {macs['quant_proj_macs']:>14,}")
    print(f"  FP attention (QK/AV):      {macs['fp_attn_macs']:>14,}")
    print(f"  LM head (FP):              {macs['lm_head_macs']:>14,}")
    print(f"  ────────────────────────────────────────")
    print(f"  TOTAL:                     {macs['total_macs']:>14,}")
    print(f"  Projection weight params:  {macs['proj_weight_params']:>14,}")
    print(f"  Embed params (tied head):  {macs['embed_params']:>14,}")
    print(f"  Quant modules (scales):    {macs['n_quant_modules']:>14,}")
    print()


def print_compute_table(rows: list[dict], fp_compute: float):
    print("─── Compute energy / token (Horowitz 45nm op model) ───")
    print(
        f"  {'Baseline':<16s} {'pJ/projMAC':>10s} {'proj nJ':>10s} "
        f"{'FP nJ':>10s} {'total nJ':>10s} {'vs FP':>8s}"
    )
    print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
    for r in rows:
        fp_part = r["compute_fp_attn_nJ"] + r["compute_lm_nJ"]
        vs = 100.0 * r["compute_total_nJ"] / fp_compute
        print(
            f"  {r['baseline']:<16s} {r['pj_per_proj_mac']:>10.2f} "
            f"{r['compute_proj_nJ']:>10.1f} {fp_part:>10.1f} "
            f"{r['compute_total_nJ']:>10.1f} {vs:>7.1f}%"
        )
    print()


def print_memory_table(rows: list[dict], fp_mem: float):
    print(
        f"─── Memory energy / token (DRAM {rows[0]['pj_per_bit']:.1f} pJ/bit, "
        f"decode re-read weights) ───"
    )
    print(
        f"  {'Baseline':<16s} {'bit/w':>6s} {'proj Mbit':>10s} "
        f"{'mem nJ':>10s} {'vs FP':>8s}"
    )
    print(f"  {'─'*16} {'─'*6} {'─'*10} {'─'*10} {'─'*8}")
    for r in rows:
        vs = 100.0 * r["memory_total_nJ"] / fp_mem
        print(
            f"  {r['baseline']:<16s} {r['bits_per_weight']:>6.2f} "
            f"{r['proj_weight_Mbit']:>10.3f} {r['memory_total_nJ']:>10.1f} {vs:>7.1f}%"
        )
    print()


def print_total_table(rows: list[dict], fp_total: float):
    print("─── Total energy / token (compute + memory) ───")
    print(
        f"  {'Baseline':<16s} {'compute':>10s} {'memory':>10s} "
        f"{'TOTAL nJ':>10s} {'vs FP':>8s}  op"
    )
    print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*10} {'─'*8}  {'─'*28}")
    for r in rows:
        vs = 100.0 * r["total_nJ"] / fp_total
        print(
            f"  {r['baseline']:<16s} {r['compute_total_nJ']:>10.1f} "
            f"{r['memory_total_nJ']:>10.1f} {r['total_nJ']:>10.1f} "
            f"{vs:>7.1f}%  {r['op']}"
        )
    print()


def print_summary(rows: list[dict], fp_total: float):
    print("=" * 88)
    print(f"{'SUMMARY: Compute + Memory Energy (10M-scale)':^88s}")
    print("=" * 88)
    print(
        f"  {'Baseline':<16s} {'bits':>5s} {'comp nJ':>9s} {'mem nJ':>9s} "
        f"{'total':>9s} {'%FP':>7s} {'×save':>7s}"
    )
    print(f"  {'─'*16} {'─'*5} {'─'*9} {'─'*9} {'─'*9} {'─'*7} {'─'*7}")
    for r in rows:
        pct = 100.0 * r["total_nJ"] / fp_total
        save = fp_total / r["total_nJ"] if r["total_nJ"] > 0 else float("inf")
        print(
            f"  {r['baseline']:<16s} {r['bits_per_weight']:>5.2f} "
            f"{r['compute_total_nJ']:>9.1f} {r['memory_total_nJ']:>9.1f} "
            f"{r['total_nJ']:>9.1f} {pct:>6.1f}% {save:>6.2f}×"
        )
    print("=" * 88)
    print(
        "Notes: compute uses split MACs (quant proj vs FP attn/LM). "
        "Memory uses DRAM bit energy; embeddings counted FP32 for all."
    )
    print(
        "BitNet bits=1.58 is ideal ternary entropy; use --packed-bitnet for 2-bit packing."
    )
    print()


def save_csv(path: str, rows: list[dict]):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {path}")


def save_json(path: str, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")


def benchmark_throughput(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    n_warmup: int = 5,
    n_iter: int = 50,
    device: str = "cpu",
):
    model = model.to(device)
    model.eval()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            model(input_ids)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed * 1000 / n_iter
    tok_s = batch_size * seq_len / (ms / 1000)
    return tok_s, ms


def main():
    parser = argparse.ArgumentParser(description="Compute + memory energy comparison")
    parser.add_argument("--baselines", nargs="+", default=ALL_BASELINES, choices=ALL_BASELINES)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    parser.add_argument("--ffn-dim", type=int, default=DEFAULT_CONFIG["ffn_dim"])
    parser.add_argument("--num-heads", type=int, default=DEFAULT_CONFIG["num_heads"])
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG["max_seq_len"])
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_CONFIG["vocab_size"])
    parser.add_argument(
        "--packed-bitnet",
        action="store_true",
        help="Count BitNet weights as 2-bit packed (not ideal 1.58)",
    )
    parser.add_argument(
        "--pj-per-bit",
        type=float,
        default=DRAM_PJ_PER_BIT,
        help=f"Memory energy pJ/bit (default DRAM {DRAM_PJ_PER_BIT})",
    )
    parser.add_argument(
        "--sram",
        action="store_true",
        help=f"Use on-chip SRAM bit energy ({SRAM_PJ_PER_BIT} pJ/bit) instead of DRAM",
    )
    parser.add_argument("--csv", type=str, default=None, help="Write CSV results")
    parser.add_argument("--json", type=str, default=None, help="Write JSON results")
    parser.add_argument(
        "--throughput",
        action="store_true",
        help="Also run training-style FP forward throughput (not specialized kernels)",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    args = parser.parse_args()

    pj_bit = SRAM_PJ_PER_BIT if args.sram else args.pj_per_bit

    cfg = dict(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.seq_len,
        tie_weights=True,
    )

    macs = count_macs_per_token(cfg)
    n_params_proj = macs["proj_weight_params"]
    n_params_total = n_params_proj + macs["embed_params"]

    print(f"Model (~10M class): dim={cfg['hidden_dim']}, layers={cfg['num_layers']}, "
          f"heads={cfg['num_heads']}, ffn={cfg['ffn_dim']}, vocab={cfg['vocab_size']}, "
          f"T={cfg['max_seq_len']}")
    print(f"Params: projections={n_params_proj:,}  embed={macs['embed_params']:,}  "
          f"total≈{n_params_total:,}")
    print(f"Memory model: {'SRAM' if args.sram else 'DRAM'} @ {pj_bit} pJ/bit")
    print(f"BitNet weight bits: "
          f"{'2.0 packed' if args.packed_bitnet else '1.58 ideal'}")
    print()

    print_mac_split(macs)

    def _row(baseline: str) -> dict:
        packed = args.packed_bitnet if baseline == "bitnet" else False
        comp = compute_energy_nj(baseline, macs)
        mem = memory_energy_nj(
            baseline, macs, use_packed_bitnet=packed, pj_per_bit=pj_bit
        )
        return {
            "baseline": baseline,
            "op": BASELINE_OP_DESC[baseline],
            **comp,
            **mem,
            "total_nJ": comp["compute_total_nJ"] + mem["memory_total_nJ"],
        }

    rows = [_row(b) for b in args.baselines]
    fp_row = next((r for r in rows if r["baseline"] == "full_precision"), _row("full_precision"))

    print_compute_table(rows, fp_row["compute_total_nJ"])
    print_memory_table(rows, fp_row["memory_total_nJ"])
    print_total_table(rows, fp_row["total_nJ"])
    print_summary(rows, fp_row["total_nJ"])

    # Secondary: SRAM sensitivity for memory-only
    if not args.sram:
        print("─── Sensitivity: memory energy if weights stay in SRAM ───")
        sram_rows = []
        for b in args.baselines:
            mem = memory_energy_nj(
                b, macs,
                use_packed_bitnet=(args.packed_bitnet if b == "bitnet" else False),
                pj_per_bit=SRAM_PJ_PER_BIT,
            )
            comp = compute_energy_nj(b, macs)
            sram_rows.append({
                "baseline": b,
                "memory_total_nJ": mem["memory_total_nJ"],
                "total_nJ": comp["compute_total_nJ"] + mem["memory_total_nJ"],
            })
        fp_s = next(r for r in sram_rows if r["baseline"] == "full_precision")
        print(f"  {'Baseline':<16s} {'mem nJ':>10s} {'total nJ':>10s} {'vs FP':>8s}")
        for r in sram_rows:
            print(
                f"  {r['baseline']:<16s} {r['memory_total_nJ']:>10.1f} "
                f"{r['total_nJ']:>10.1f} {100*r['total_nJ']/fp_s['total_nJ']:>7.1f}%"
            )
        print()

    payload = {
        "config": cfg,
        "macs": {k: v for k, v in macs.items() if k not in ("proj_mac_breakdown", "param_breakdown")},
        "proj_mac_breakdown": macs["proj_mac_breakdown"],
        "pj_per_bit": pj_bit,
        "packed_bitnet": args.packed_bitnet,
        "rows": rows,
    }

    if args.csv:
        save_csv(args.csv, rows)
    if args.json:
        save_json(args.json, payload)

    if args.throughput:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
        if device == "cpu":
            torch.set_num_threads(n_threads)
        print(f"─── Optional throughput ({device}, training-style forward) ───")
        for name in args.baselines:
            if name not in ("full_precision", "bitnet", "uniform_2bit", "fixed_c_025",
                            "fixed_c_05", "learned_c", "heterogeneous"):
                continue
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
            for bs in args.batch_sizes:
                tok_s, ms = benchmark_throughput(
                    model, bs, min(cfg["max_seq_len"], 128), cfg["vocab_size"],
                    n_iter=args.iterations, device=device,
                )
                print(f"    bs={bs}: {tok_s:,.0f} tok/s  ({ms:.2f} ms/fwd)")
            del model


if __name__ == "__main__":
    main()
