import argparse
import csv
import json
from datetime import datetime, timezone

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from train_kaggle import build_model

SEQ_LEN = 512
BATCH_SIZE = 4
MAX_STORIES = 2500

ALL_BASELINES = [
    "full_precision",
    "bitnet",
    "uniform_2bit",
    "fixed_c_025",
    "fixed_c_05",
    "learned_c",
]

MODEL_CONFIG = dict(
    vocab_size=4096,
    hidden_dim=256,
    num_layers=8,
    num_heads=8,
    ffn_dim=1024,
    max_seq_len=SEQ_LEN,
    tie_weights=True,
)


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def tokenize_valid(data_path: str, tokenizer: Tokenizer, max_stories: int = -1):
    all_ids = []
    num_stories = 0
    with open(data_path, "r") as f:
        for line in f:
            if max_stories > 0 and num_stories >= max_stories:
                break
            ids = tokenizer.encode(line.strip()).ids
            all_ids.extend(ids)
            num_stories += 1
    tokens = np.array(all_ids, dtype=np.int64)
    print(f"Tokenized {num_stories} stories -> {len(tokens):,} tokens")
    return torch.from_numpy(tokens)


class ValidDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_sequences = len(tokens) // (seq_len + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def eval_ppl(model, loader):
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            out = model(input_ids, labels=labels)
            loss = out["loss"]
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    avg_loss = total_loss / total_tokens
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all baselines on TinyStories-valid"
    )
    parser.add_argument("--tokenizer", default="tetranet_tokenizer.json")
    parser.add_argument(
        "--valid-data", default="tinystories/TinyStoriesV2-GPT4-valid.txt"
    )
    parser.add_argument("--max-stories", type=int, default=MAX_STORIES)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=ALL_BASELINES,
        choices=ALL_BASELINES,
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--json", default="ppl_10m.json", help="Write JSON results")
    parser.add_argument("--csv", default="ppl_10m.csv", help="Write CSV results")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA unavailable — using CPU")

    tokenizer = load_tokenizer(args.tokenizer)
    tokens = tokenize_valid(args.valid_data, tokenizer, args.max_stories)
    dataset = ValidDataset(tokens, SEQ_LEN)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    print(f"Validation sequences: {len(dataset):,}  device={device}\n")

    results = {}
    for name in args.baselines:
        ckpt_path = f"checkpoint_{name}_final.pt"
        print(f"─── {name} ({ckpt_path}) ───", flush=True)
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        except FileNotFoundError:
            print(f"  SKIP — checkpoint not found: {ckpt_path}")
            continue

        model = build_model(baseline=name, **MODEL_CONFIG)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        # Move batches to device inside eval
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for input_ids, labels in loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                out = model(input_ids, labels=labels)
                loss = out["loss"]
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = float(torch.exp(torch.tensor(avg_loss)).item())

        entry = {
            "baseline": name,
            "checkpoint": ckpt_path,
            "loss": avg_loss,
            "ppl": ppl,
            "tokens_eval": total_tokens,
        }

        if name == "learned_c":
            c_vals = model.get_c_values()
            snapped = {"0.25": 0, "0.50": 0, "other": 0}
            for v in c_vals.values():
                if abs(v - 0.25) < 0.01:
                    snapped["0.25"] += 1
                elif abs(v - 0.50) < 0.01:
                    snapped["0.50"] += 1
                else:
                    snapped["other"] += 1
            entry["c_snapped"] = snapped
            entry["n_c"] = len(c_vals)
            print(f"  c params: {len(c_vals)}  snapped {snapped}")

        results[name] = entry
        print(f"  loss={avg_loss:.4f}  ppl={ppl:.2f}")
        print()
        del model

    fp_ppl = results.get("full_precision", {}).get("ppl")
    bn_ppl = results.get("bitnet", {}).get("ppl")

    rows = []
    for name in args.baselines:
        if name not in results:
            continue
        r = results[name]
        delta_fp = (r["ppl"] - fp_ppl) if fp_ppl is not None else None
        delta_bn = (r["ppl"] - bn_ppl) if bn_ppl is not None else None
        gap_closed = None
        if (
            fp_ppl is not None
            and bn_ppl is not None
            and bn_ppl != fp_ppl
            and name not in ("full_precision", "bitnet")
        ):
            # fraction of BitNet→FP gap recovered (positive = better than BitNet toward FP)
            gap_closed = (bn_ppl - r["ppl"]) / (bn_ppl - fp_ppl)
        row = {
            **r,
            "delta_vs_fp": delta_fp,
            "delta_vs_bitnet": delta_bn,
            "gap_closed_vs_bitnet": gap_closed,
            "beats_bitnet": (delta_bn is not None and delta_bn < 0),
        }
        rows.append(row)
        results[name] = row

    # Summary table
    print("=" * 78)
    print(f"{'Baseline':<18s} {'Loss':>8s} {'PPL':>8s} {'Δ vs FP':>10s} {'Δ vs BitNet':>12s}")
    print("-" * 78)
    for row in rows:
        d_fp = f"{row['delta_vs_fp']:+.2f}" if row["delta_vs_fp"] is not None else "—"
        d_bn = (
            f"{row['delta_vs_bitnet']:+.2f}"
            if row["delta_vs_bitnet"] is not None
            else "—"
        )
        mark = " *" if row.get("beats_bitnet") else ""
        print(
            f"{row['baseline']:<18s} {row['loss']:>8.4f} {row['ppl']:>8.2f} "
            f"{d_fp:>10s} {d_bn:>12s}{mark}"
        )
    print("=" * 78)
    print("* = lower PPL than BitNet (better)")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": MODEL_CONFIG,
        "eval": {
            "valid_data": args.valid_data,
            "tokenizer": args.tokenizer,
            "max_stories": args.max_stories,
            "seq_len": SEQ_LEN,
            "batch_size": args.batch_size,
            "device": str(device),
            "n_sequences": len(dataset),
            "n_tokens": int(tokens.numel()),
        },
        "results": results,
        "rows": rows,
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.json}")

    if args.csv:
        fieldnames = [
            "baseline",
            "loss",
            "ppl",
            "delta_vs_fp",
            "delta_vs_bitnet",
            "gap_closed_vs_bitnet",
            "beats_bitnet",
            "tokens_eval",
            "checkpoint",
        ]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
