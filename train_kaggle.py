"""
GPU training entry point for Kaggle Notebooks.

Self-contained: no dependency on `models/` package (Kaggle Datasets don't support
subdirectories). Only depends on model.py and quaternary.py.

Supports all 4 baselines, mixed precision, checkpoint resume (for Kaggle 9h limit).

Usage on Kaggle:
    !python train_kaggle.py --baseline fixed_c_05 --epochs 1
    !python train_kaggle.py --baseline fixed_c_025 --epochs 1
    !python train_kaggle.py --baseline fixed_c_075 --epochs 1
    !python train_kaggle.py --baseline bitnet --epochs 1
    !python train_kaggle.py --baseline uniform_2bit --epochs 1
    !python train_kaggle.py --baseline full_precision --epochs 1
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import QuaternaryLlamaForCausalLM, QuaternaryLlamaConfig
from quaternary import FixedCQuaternaryLinear, LearnedCQuaternaryLinear
from regularization import AdaptiveSnappingScheduler, compute_total_loss, multi_well_potential


# ══════════════════════════════════════════════════
#  4 Baseline Quantizer Layers
# ══════════════════════════════════════════════════


class FullPrecisionLinear(nn.Linear):
    """Standard nn.Linear — no quantization.
    Serves as the theoretical upper bound baseline."""

    def __init__(self, in_features, out_features, bias=False, **kwargs):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class BitNetTernaryFunction(torch.autograd.Function):
    """STE backward for BitNet b1.58 ternary quantization."""

    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, threshold)
        y = torch.sign(x)
        y[x.abs() < 0.5] = 0.0
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x.abs() > threshold] = 0.0
        return grad_x, None


class BitNetTernaryLinear(nn.Linear):
    """BitNet b1.58-style ternary quantization to {-1, 0, 1} with STE."""

    def __init__(self, in_features, out_features, bias=False, threshold=1.0, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
        self.threshold = threshold

    def _ternary_quantize(self, weight):
        gamma = weight.abs().mean().clamp(min=1e-8).detach()
        scaled = weight / gamma
        quantized = BitNetTernaryFunction.apply(
            scaled,
            torch.tensor(self.threshold, device=weight.device, dtype=weight.dtype),
        )
        return quantized * gamma

    def forward(self, x):
        qweight = self._ternary_quantize(self.weight)
        return F.linear(x, qweight, self.bias)


class Uniform2BitFunction(torch.autograd.Function):
    """STE backward for uniform 2-bit quantization to {-a, -a/3, a/3, a}."""

    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, threshold)
        levels = torch.tensor(
            [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=x.device, dtype=x.dtype
        )
        flat = x.view(-1, 1)
        dist = (flat - levels.unsqueeze(0)).abs()
        indices = dist.argmin(dim=1)
        y = levels[indices].view(x.shape)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x.abs() > threshold] = 0.0
        return grad_x, None


class Uniform2BitLinear(nn.Linear):
    """Uniform 2-bit quantization to 4 evenly-spaced levels with STE.

    States: {-alpha, -alpha/3, alpha/3, alpha}
    This is the naive 2-bit baseline that does NOT use power-of-two shifting.
    """

    def __init__(self, in_features, out_features, bias=False, threshold=1.0, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
        self.threshold = threshold

    def _uniform_quantize(self, weight):
        gamma = weight.abs().max().clamp(min=1e-8).detach()
        scaled = weight / gamma
        quantized = Uniform2BitFunction.apply(
            scaled,
            torch.tensor(self.threshold, device=weight.device, dtype=weight.dtype),
        )
        return quantized * gamma

    def forward(self, x):
        qweight = self._uniform_quantize(self.weight)
        return F.linear(x, qweight, self.bias)


# ══════════════════════════════════════════════════
#  Model Factory
# ══════════════════════════════════════════════════

BASELINES = {
    "full_precision": FullPrecisionLinear,
    "bitnet": BitNetTernaryLinear,
    "uniform_2bit": Uniform2BitLinear,
    "fixed_c_025": FixedCQuaternaryLinear,
    "fixed_c_05": FixedCQuaternaryLinear,
    "fixed_c_075": FixedCQuaternaryLinear,
    "learned_c": LearnedCQuaternaryLinear,
}


def build_model(
    baseline: str = "quaternary",
    vocab_size: int = 4096,
    hidden_dim: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    ffn_dim: int = 3072,
    max_seq_len: int = 512,
    initial_c: float = 0.375,
    threshold: float = 1.0,
    tie_weights: bool = True,
    rope_base: float = 10000.0,
) -> QuaternaryLlamaForCausalLM:
    if baseline not in BASELINES:
        raise ValueError(
            f"Unknown baseline: {baseline}. Choose from {list(BASELINES.keys())}"
        )

    fixed_c_map = {
        "fixed_c_025": 0.25,
        "fixed_c_05": 0.5,
        "fixed_c_075": 0.75,
    }
    if baseline in fixed_c_map:
        initial_c = fixed_c_map[baseline]

    config = QuaternaryLlamaConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
        initial_c=initial_c,
        threshold=threshold,
        tie_weights=tie_weights,
        rope_base=rope_base,
        linear_cls=BASELINES[baseline],
    )

    return QuaternaryLlamaForCausalLM(config)


# ══════════════════════════════════════════════════
#  Snapping helpers
# ══════════════════════════════════════════════════


def count_snapped(c_values: dict[str, float], wells=(0.25, 0.5), tol=0.01):
    snapped = {w: 0 for w in wells}
    for v in c_values.values():
        for w in wells:
            if abs(v - w) < tol:
                snapped[w] += 1
                break
    return snapped


# ══════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════


class TinyStoriesDataset(Dataset):
    def __init__(self, tokens_path: str, seq_len: int = 512):
        self.data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.num_sequences = len(self.data) // (seq_len + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64)
        )
        return chunk[:-1], chunk[1:]


# ══════════════════════════════════════════════════
#  Tokenizer helpers
# ══════════════════════════════════════════════════


def load_tokenizer(tokenizer_path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(tokenizer_path)


def get_vocab_size(tokenizer_path: str) -> int:
    return load_tokenizer(tokenizer_path).get_vocab_size()


def tokenize_dataset(
    data_path: str,
    tokenizer_path: str,
    output_path: str,
    max_stories: int = -1,
):
    if Path(output_path).exists():
        print(f"Found existing tokenized file: {output_path}")
        return

    tokenizer = load_tokenizer(tokenizer_path)
    all_ids = []

    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if max_stories > 0 and i >= max_stories:
                break
            ids = tokenizer.encode(line.strip()).ids
            all_ids.extend(ids)

    tokens = np.array(all_ids, dtype=np.uint16)
    tokens.tofile(output_path)

    max_id = int(tokens.max())
    assert max_id < get_vocab_size(tokenizer_path), (
        f"Token ID {max_id} exceeds vocab size. "
        "Regenerate tokenizer with larger vocab."
    )

    print(
        f"Tokenized {i+1 if max_stories < 0 else max_stories} stories "
        f"\u2192 {len(tokens):,} tokens \u2192 {output_path}"
    )


# ══════════════════════════════════════════════════
#  Checkpoint
# ══════════════════════════════════════════════════


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    tokens_seen: int,
    epoch: int,
    best_loss: float,
    args,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "epoch": epoch,
        "best_loss": best_loss,
        "args": vars(args),
    }
    torch.save(ckpt, path)
    print(f"  Checkpoint saved \u2192 {path} (step {step})")


def load_checkpoint(path: str, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return (
        ckpt["step"],
        ckpt["tokens_seen"],
        ckpt.get("epoch", 0),
        ckpt.get("best_loss", float("inf")),
    )


# ══════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} "
        f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})"
    )

    use_amp = device.type == "cuda" and args.mixed_precision
    # Auto-detect BF16 support: requires CUDA capability >= 7.0 (Volta+)
    cuda_cap = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
    bf16_supported = cuda_cap >= (7, 0)
    amp_dtype = torch.bfloat16 if (args.bf16 and bf16_supported) else torch.float16
    if use_amp:
        print(f"Mixed precision: {amp_dtype} (GPU cc {cuda_cap[0]}.{cuda_cap[1]})")

    if not Path(args.tokens_path).exists():
        print("Tokenizing TinyStories...")
        tokenize_dataset(
            args.data_path,
            args.tokenizer_path,
            args.tokens_path,
            args.max_stories,
        )

    dataset = TinyStoriesDataset(args.tokens_path, seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    total_tokens = len(dataset) * args.seq_len
    total_steps = len(loader) * args.epochs
    print(
        f"Dataset: {len(dataset):,} sequences "
        f"({total_tokens:,} tokens, {total_steps:,} steps)"
    )

    vocab_size = get_vocab_size(args.tokenizer_path)
    model = build_model(
        baseline=args.baseline,
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.seq_len,
        initial_c=args.initial_c,
        threshold=args.threshold,
        tie_weights=args.tie_weights,
    ).to(device)
    num_params = model.get_num_params()
    print(f"Model: {num_params:,} params")

    has_snap = args.baseline == "learned_c"

    if has_snap:
        param_groups = [
            {
                "params": [p for n, p in model.named_parameters() if not n.endswith(".c")],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in model.named_parameters() if n.endswith(".c")],
                "lr": args.c_lr,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        snap_scheduler = AdaptiveSnappingScheduler(
            alpha=args.alpha, snap_start=args.snap_start
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        snap_scheduler = None

    scaler = torch.amp.GradScaler(
        device.type, enabled=(use_amp and amp_dtype == torch.float16)
    )

    start_step = 0
    tokens_seen = 0
    start_epoch = 0
    best_loss = float("inf")

    if args.resume and Path(args.checkpoint_path).exists():
        print(f"Resuming from {args.checkpoint_path}...")
        start_step, tokens_seen, start_epoch, best_loss = load_checkpoint(
            args.checkpoint_path, model, optimizer, scaler, device
        )
        print(
            f"  Resumed at step {start_step}, "
            f"epoch {start_epoch}, tokens_seen {tokens_seen:,}"
        )

    log_path = args.log_path
    write_header = not Path(log_path).exists()
    log_file = open(log_path, "a", newline="")
    csv_writer = csv.writer(log_file)
    if write_header:
        cols = [
            "step",
            "epoch",
            "progress",
            "task_loss",
            "lr",
            "grad_norm",
        ]
        if has_snap:
            cols += ["total_loss", "lambda", "penalty", "n_snapped_025", "n_snapped_05", "c_values_json"]
        csv_writer.writerow(cols)

    print(
        f"Starting {args.baseline} training "
        f"(lr={args.lr}, batch={args.batch_size}, epochs={args.epochs})"
    )
    start_time = time.time()

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (input_ids, labels) in enumerate(loader):
            step = start_step + batch_idx + epoch * len(loader)
            tokens_seen += input_ids.numel()
            progress = min(tokens_seen / (total_tokens * args.epochs), 1.0)

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                out = model(input_ids, labels=labels)
                task_loss = out["loss"]

            if has_snap and progress > args.snap_start:
                total_loss = compute_total_loss(
                    task_loss, model, progress, snap_scheduler
                )
                penalty = multi_well_potential(model).detach().item()
                lambda_val = snap_scheduler.get_lambda(
                    progress, task_loss.detach()
                )
            else:
                total_loss = task_loss
                penalty = 0.0
                lambda_val = 0.0

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )

                if use_amp and amp_dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
            else:
                grad_norm = torch.tensor(0.0)

            epoch_loss += task_loss.item()
            epoch_steps += 1

            if step % args.log_interval == 0:
                elapsed = time.time() - start_time

                row = [
                    step,
                    epoch,
                    f"{progress:.4f}",
                    f"{task_loss.item():.4f}",
                    f"{optimizer.param_groups[0]['lr']:.2e}",
                    f"{grad_norm.item():.4f}",
                ]
                if has_snap:
                    c_vals = model.get_c_values()
                    snapped = count_snapped(c_vals)
                    row += [
                        f"{total_loss.item():.4f}",
                        f"{lambda_val:.4f}",
                        f"{penalty:.4f}",
                        snapped[0.25],
                        snapped[0.5],
                        json.dumps(c_vals),
                    ]
                csv_writer.writerow(row)
                log_file.flush()

                if task_loss.item() < best_loss:
                    best_loss = task_loss.item()

                speed = tokens_seen / elapsed
                eta_sec = (
                    (total_tokens * args.epochs - tokens_seen) / speed
                    if speed > 0
                    else 0
                )
                eta_h = eta_sec / 3600

                print(
                    f"[{args.baseline[:4]}] "
                    f"step {step:>7} | "
                    f"epoch {epoch} | "
                    f"{elapsed / 3600:.1f}h | "
                    f"progress {progress:.3f} | "
                    f"loss {task_loss.item():.3f} | "
                    f"tok/s {speed:,.0f} | "
                    f"ETA {eta_h:.1f}h"
                )

            if step > 0 and step % args.ckpt_interval == 0:
                save_checkpoint(
                    args.checkpoint_path,
                    model,
                    optimizer,
                    scaler,
                    step,
                    tokens_seen,
                    epoch,
                    best_loss,
                    args,
                )

        save_checkpoint(
            args.checkpoint_path,
            model,
            optimizer,
            scaler,
            step,
            tokens_seen,
            epoch,
            best_loss,
            args,
        )
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch} complete \u2014 avg loss {avg_loss:.4f}")

    save_checkpoint(
        args.checkpoint_path.replace(".pt", "_final.pt"),
        model,
        optimizer,
        scaler,
        step,
        tokens_seen,
        args.epochs - 1,
        best_loss,
        args,
    )

    log_file.close()
    print(f"\nDone! Best loss: {best_loss:.4f}")


# ══════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetranet \u2014 Kaggle GPU training")

    parser.add_argument(
        "--baseline",
        default="fixed_c_05",
        choices=list(BASELINES.keys()),
        help="Which baseline to train",
    )

    parser.add_argument(
        "--data-path",
        default="/kaggle/input/tinystories/TinyStories-train.txt",
    )
    parser.add_argument("--tokenizer-path", default="./tetranet_tokenizer.json")
    parser.add_argument("--tokens-path", default="./train_tokens.bin")
    parser.add_argument(
        "--max-stories", type=int, default=500000, help="Limit stories (-1 = all, default 500K)"
    )

    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--ffn-dim", type=int, default=3072)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--tie-weights", action="store_true", default=True)
    parser.add_argument("--initial-c", type=float, default=0.375)
    parser.add_argument("--threshold", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision")
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use BF16 instead of FP16 (preferred on Ampere+)",
    )

    parser.add_argument("--checkpoint-path", default="./model_checkpoint.pt")
    parser.add_argument("--log-path", default="./training_log.csv")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if exists",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
    )

    # ── Snapping args (only used with --baseline learned_c) ──
    parser.add_argument(
        "--c-lr",
        type=float,
        default=0.003,
        help="Learning rate for c parameters (learned_c only)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Snapping penalty strength (learned_c only)",
    )
    parser.add_argument(
        "--snap-start",
        type=float,
        default=0.4,
        help="Progress threshold to activate snapping penalty (0-1, learned_c only)",
    )

    parser.add_argument("--num-workers", type=int, default=2)

    args = parser.parse_args()
    train(args)
