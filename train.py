import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from model import QuaternaryLlamaForCausalLM, QuaternaryLlamaConfig
from regularization import AdaptiveSnappingScheduler, compute_total_loss, multi_well_potential
from quaternary import QBitLinearQuaternary


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
        return chunk[:-1], chunk[1:]  # input_ids, labels


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run `python train_tokenizer.py` first."
        )
    return Tokenizer.from_file(tokenizer_path)


def get_vocab_size(tokenizer_path: str) -> int:
    return load_tokenizer(tokenizer_path).get_vocab_size()


def preprocess(
    data_path: str,
    tokenizer_path: str,
    output_path: str,
    num_stories: int = 10000,
):
    tokenizer = load_tokenizer(tokenizer_path)

    all_ids = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_stories:
                break
            ids = tokenizer.encode(line.strip()).ids
            all_ids.extend(ids)

    tokens = np.array(all_ids, dtype=np.uint16)
    max_id = int(tokens.max())
    assert max_id < 4096, (
        f"Token ID {max_id} exceeds vocab_size 4096. "
        "Regenerate tokenizer with smaller vocab or increase model vocab_size."
    )
    tokens.tofile(output_path)  # raw binary, no header
    print(f"Pre-tokenized {num_stories} stories → {len(tokens):,} tokens → {output_path}")


def count_snapped(c_values, wells=(0.25, 0.5), tol=0.01):
    snapped = {w: 0 for w in wells}
    for v in c_values.values():
        for w in wells:
            if abs(v - w) < tol:
                snapped[w] += 1
                break
    return snapped


def train(args):
    device = torch.device("cpu")

    # ── Pre-tokenize if needed ──
    tokens_path = args.tokens_path
    if not Path(tokens_path).exists():
        print("Pre-tokenizing TinyStories...")
        preprocess(args.data_path, args.tokenizer_path, tokens_path, args.num_stories)

    # ── Dataset ──
    dataset = TinyStoriesDataset(tokens_path, seq_len=args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    total_tokens = len(dataset) * args.seq_len
    print(f"Dataset: {len(dataset)} sequences ({total_tokens:,} tokens)")

    # ── Model ──
    config = QuaternaryLlamaConfig(
        vocab_size=get_vocab_size(args.tokenizer_path),
        hidden_dim=256,
        num_layers=8,
        num_heads=8,
        ffn_dim=1024,
        max_seq_len=args.seq_len,
        initial_c=0.375,
        threshold=1.0,
        tie_weights=True,
    )
    model = QuaternaryLlamaForCausalLM(config).to(device)
    num_params = model.get_num_params()
    print(f"Model: {num_params:,} params")

    # ── Optimizer ──
    # Separate learning rate for c parameters (higher to overcome AdamW normalization)
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
    )

    snap_scheduler = AdaptiveSnappingScheduler(
        alpha=args.alpha, snap_start=args.snap_start
    )

    # ── CSV logging ──
    log_file = open(args.log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "step", "progress", "task_loss", "total_loss",
        "lambda", "penalty", "lr",
        "n_snapped_025", "n_snapped_05",
        "c_grad_mean", "c_grad_max",
    ])

    # ── Training loop ──
    tokens_seen = 0
    step = 0
    num_snap_params = len(
        [m for m in model.modules() if isinstance(m, QBitLinearQuaternary)]
    )

    print(f"Starting training (alpha={args.alpha}, snap_start={args.snap_start}, c_lr={args.c_lr})")
    start_time = time.time()

    model.train()
    for epoch in range(1):
        for input_ids, labels in loader:
            step += 1
            tokens_seen += input_ids.numel()
            progress = min(tokens_seen / total_tokens, 1.0)

            out = model(input_ids, labels=labels)
            task_loss = out["loss"]

            # Phase 2: add snapping penalty
            if progress > args.snap_start:
                total_loss = compute_total_loss(
                    task_loss, model, progress, snap_scheduler
                )
                penalty = multi_well_potential(model).detach().item()
                lambda_val = snap_scheduler.get_lambda(progress, task_loss.detach())
            else:
                total_loss = task_loss
                penalty = 0.0
                lambda_val = 0.0

            total_loss.backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            # ── Logging ──
            if step % args.log_interval == 0:
                c_vals = model.get_c_values()
                snapped = count_snapped(c_vals)
                elapsed = time.time() - start_time

                # Log c gradient stats (only available before optimizer.step)
                c_grads = [
                    p.grad.item()
                    for name, p in model.named_parameters()
                    if name.endswith(".c") and p.grad is not None
                ]
                c_grad_mean = sum(c_grads) / len(c_grads) if c_grads else 0.0
                c_grad_max = max(abs(g) for g in c_grads) if c_grads else 0.0

                csv_writer.writerow([
                    step,
                    f"{progress:.4f}",
                    f"{task_loss.item():.4f}",
                    f"{total_loss.item():.4f}",
                    f"{lambda_val:.4f}",
                    f"{penalty:.6f}",
                    f"{args.lr:.2e}",
                    snapped[0.25],
                    snapped[0.5],
                    f"{c_grad_mean:.4f}",
                    f"{c_grad_max:.4f}",
                ])
                log_file.flush()

                print(
                    f"step {step:>5} | "
                    f"{elapsed / 60:>5.1f}min | "
                    f"progress {progress:.3f} | "
                    f"loss {task_loss.item():.3f} | "
                    f"λ {lambda_val:.3f} | "
                    f"snapped {snapped[0.25]}+{snapped[0.5]}/{num_snap_params} | "
                    f"cg_mean {c_grad_mean:+.2f} cg_max {c_grad_max:.2f}"
                )

    # ── Save checkpoint ──
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "optimizer_state_dict": optimizer.state_dict(),
        "c_values": model.get_c_values(),
        "step": step,
        "tokens_seen": tokens_seen,
    }
    torch.save(checkpoint, args.checkpoint_path)
    print(f"\nCheckpoint saved to {args.checkpoint_path}")

    # ── Final c report ──
    c_vals = model.get_c_values()
    snapped = count_snapped(c_vals)
    print(f"Final c distribution: {snapped[0.25]} at 0.25, {snapped[0.5]} at 0.5")
    for name, c_val in sorted(c_vals.items()):
        print(f"  {name}: {c_val:.4f}")

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./tinystories/TinyStories-train.txt")
    parser.add_argument("--tokenizer-path", default="./tetranet_tokenizer.json")
    parser.add_argument("--tokens-path", default="./train_tokens.bin")
    parser.add_argument("--checkpoint-path", default="./model_final.pt")
    parser.add_argument("--log-path", default="./training_log.csv")
    parser.add_argument("--num-stories", type=int, default=10000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--c-lr", type=float, default=0.003,
                        help="Learning rate for c parameters (higher to enable snapping)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--snap-start", type=float, default=0.4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()
    train(args)
