"""
GPU training entry point for Kaggle Notebooks.

Supports all 4 baselines, mixed precision, checkpoint resume (for Kaggle 9h limit),
and auto-detects T4/P100/A100 GPU.

Usage on Kaggle:
    !python train_kaggle.py --baseline full_precision --epochs 1
    !python train_kaggle.py --baseline bitnet --epochs 1
    !python train_kaggle.py --baseline uniform_2bit --epochs 1
    !python train_kaggle.py --baseline quaternary --epochs 1
"""

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import build_model, BASELINES
from regularization import AdaptiveSnappingScheduler, compute_total_loss, multi_well_potential
from quaternary import QBitLinearQuaternary


# ──────────────────────────────────────────────
#  Dataset — memory-mapped token .bin file
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
#  Tokenizer loading
# ──────────────────────────────────────────────

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
    """Tokenize a text file of TinyStories into a raw binary .bin file."""
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

    print(f"Tokenized {i+1 if max_stories < 0 else max_stories} stories "
          f"→ {len(tokens):,} tokens → {output_path}")


# ──────────────────────────────────────────────
#  Snapping helpers
# ──────────────────────────────────────────────

def count_snapped(c_values, wells=(0.25, 0.5), tol=0.01):
    snapped = {w: 0 for w in wells}
    for v in c_values.values():
        for w in wells:
            if abs(v - w) < tol:
                snapped[w] += 1
                break
    return snapped


# ──────────────────────────────────────────────
#  Checkpoint resume
# ──────────────────────────────────────────────

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
    print(f"  Checkpoint saved → {path} (step {step})")


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


# ──────────────────────────────────────────────
#  Main training
# ──────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # ── Mixed precision ──
    use_amp = device.type == "cuda" and args.mixed_precision
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    if use_amp:
        print(f"Mixed precision: {amp_dtype}")

    # ── Tokenize on first run ──
    if not Path(args.tokens_path).exists():
        print("Tokenizing TinyStories...")
        tokenize_dataset(
            args.data_path,
            args.tokenizer_path,
            args.tokens_path,
            args.max_stories,
        )

    # ── Dataset ──
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
    print(f"Dataset: {len(dataset):,} sequences ({total_tokens:,} tokens, {total_steps:,} steps)")

    # ── Model ──
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

    # ── Optimizer ──
    has_c_params = args.baseline == "quaternary"
    if has_c_params:
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
    else:
        param_groups = [
            {
                "params": model.parameters(),
                "lr": args.lr,
            }
        ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    # Gradient scaler for FP16 (not needed for BF16)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # Snapping scheduler (Quaternary only)
    snap_scheduler = AdaptiveSnappingScheduler(
        alpha=args.alpha,
        snap_start=args.snap_start,
    )

    # ── Checkpoint resume ──
    start_step = 0
    tokens_seen = 0
    start_epoch = 0
    best_loss = float("inf")

    if args.resume and Path(args.checkpoint_path).exists():
        print(f"Resuming from {args.checkpoint_path}...")
        start_step, tokens_seen, start_epoch, best_loss = load_checkpoint(
            args.checkpoint_path, model, optimizer, scaler, device
        )
        print(f"  Resumed at step {start_step}, epoch {start_epoch}, tokens_seen {tokens_seen:,}")

    # ── CSV logging ──
    log_path = args.log_path
    write_header = not Path(log_path).exists()
    log_file = open(log_path, "a", newline="")
    csv_writer = csv.writer(log_file)
    if write_header:
        csv_writer.writerow([
            "step", "epoch", "progress", "task_loss", "total_loss",
            "lambda", "penalty", "lr", "grad_norm",
            "n_snapped_025", "n_snapped_05",
            "c_grad_mean", "c_grad_max",
        ])

    # ── Training loop ──
    num_snap_params = len(
        [m for m in model.modules() if isinstance(m, QBitLinearQuaternary)]
    ) if has_c_params else 0

    print(f"Starting {args.baseline} training "
          f"(lr={args.lr}, batch={args.batch_size}, epochs={args.epochs})")
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

            # ── Forward ──
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids, labels=labels)
                task_loss = out["loss"]

                if has_c_params and progress > args.snap_start:
                    total_loss = compute_total_loss(
                        task_loss, model, progress, snap_scheduler
                    )
                    penalty = multi_well_potential(model).detach().item()
                    lambda_val = snap_scheduler.get_lambda(progress, task_loss.detach())
                else:
                    total_loss = task_loss
                    penalty = 0.0
                    lambda_val = 0.0

            # ── Backward ──
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

            # ── Logging ──
            if step % args.log_interval == 0:
                c_vals = model.get_c_values()
                snapped = count_snapped(c_vals) if has_c_params else {}
                elapsed = time.time() - start_time

                c_grads = [
                    p.grad.item()
                    for n, p in model.named_parameters()
                    if n.endswith(".c") and p.grad is not None
                ]
                c_grad_mean = sum(c_grads) / len(c_grads) if c_grads else 0.0
                c_grad_max = max(abs(g) for g in c_grads) if c_grads else 0.0

                csv_writer.writerow([
                    step,
                    epoch,
                    f"{progress:.4f}",
                    f"{task_loss.item():.4f}",
                    f"{total_loss.item():.4f}",
                    f"{lambda_val:.4f}",
                    f"{penalty:.6f}",
                    f"{optimizer.param_groups[0]['lr']:.2e}",
                    f"{grad_norm.item():.4f}",
                    snapped.get(0.25, 0),
                    snapped.get(0.5, 0),
                    f"{c_grad_mean:.4f}",
                    f"{c_grad_max:.4f}",
                ])
                log_file.flush()

                # Track best loss
                if task_loss.item() < best_loss:
                    best_loss = task_loss.item()

                speed = tokens_seen / elapsed
                eta_sec = (total_tokens * args.epochs - tokens_seen) / speed if speed > 0 else 0
                eta_h = eta_sec / 3600

                snap_info = (f"| λ {lambda_val:.3f} | snapped {snapped.get(0.25, 0)}+{snapped.get(0.5, 0)}/{num_snap_params}"
                             if has_c_params else "")
                print(
                    f"[{args.baseline[:4]}] "
                    f"step {step:>7} | "
                    f"epoch {epoch} | "
                    f"{elapsed / 3600:.1f}h | "
                    f"progress {progress:.3f} | "
                    f"loss {task_loss.item():.3f} | "
                    f"tok/s {speed:,.0f} | "
                    f"ETA {eta_h:.1f}h"
                    f"{snap_info}"
                )

            # ── Periodic checkpoint ──
            if step > 0 and step % args.ckpt_interval == 0:
                save_checkpoint(
                    args.checkpoint_path,
                    model, optimizer, scaler,
                    step, tokens_seen, epoch, best_loss, args,
                )

        # End of epoch checkpoint
        save_checkpoint(
            args.checkpoint_path,
            model, optimizer, scaler,
            step, tokens_seen, epoch, best_loss, args,
        )
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch} complete — avg loss {avg_loss:.4f}")

    # ── Final save ──
    save_checkpoint(
        args.checkpoint_path.replace(".pt", "_final.pt"),
        model, optimizer, scaler,
        step, tokens_seen, args.epochs - 1, best_loss, args,
    )

    # ── Final c report ──
    if has_c_params:
        c_vals = model.get_c_values()
        snapped = count_snapped(c_vals)
        print(f"\nFinal c: {snapped[0.25]} at 0.25, {snapped[0.5]} at 0.5")
        for name, val in sorted(c_vals.items()):
            print(f"  {name}: {val:.4f}")

    log_file.close()
    print(f"\nDone! Best loss: {best_loss:.4f}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tetranet — Kaggle GPU training")

    # Baseline
    parser.add_argument("--baseline", default="quaternary",
                        choices=list(BASELINES.keys()),
                        help="Which baseline to train")

    # Data
    parser.add_argument("--data-path", default="/kaggle/input/tinystories/TinyStories-train.txt")
    parser.add_argument("--tokenizer-path", default="./tetranet_tokenizer.json")
    parser.add_argument("--tokens-path", default="./train_tokens.bin")
    parser.add_argument("--max-stories", type=int, default=-1,
                        help="Limit stories for tokenization (-1 = all)")

    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--ffn-dim", type=int, default=3072)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--tie-weights", action="store_true", default=True)
    parser.add_argument("--initial-c", type=float, default=0.375)
    parser.add_argument("--threshold", type=float, default=1.0)

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--c-lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Snapping (Quaternary only)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--snap-start", type=float, default=0.4)

    # Mixed precision
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use BF16 instead of FP16 (preferred on Ampere+)")

    # Checkpoint / resume
    parser.add_argument("--checkpoint-path", default="./model_checkpoint.pt")
    parser.add_argument("--log-path", default="./training_log.csv")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from checkpoint if exists")
    parser.add_argument("--ckpt-interval", type=int, default=5000,
                        help="Save checkpoint every N steps")

    # Performance
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)

    args = parser.parse_args()
    train(args)
