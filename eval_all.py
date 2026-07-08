import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
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
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    tokens = tokenize_valid(args.valid_data, tokenizer, args.max_stories)
    dataset = ValidDataset(tokens, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"Validation sequences: {len(dataset):,}\n")

    results = {}
    for name in args.baselines:
        ckpt_path = f"checkpoint_{name}_final.pt"
        print(f"─── {name} ({ckpt_path}) ───", flush=True)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            print(f"  SKIP — checkpoint not found: {ckpt_path}")
            continue

        model = build_model(baseline=name, **MODEL_CONFIG)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        loss, ppl = eval_ppl(model, loader)
        results[name] = {"loss": loss, "ppl": ppl}
        print(f"  loss={loss:.4f}  ppl={ppl:.2f}")

        if name == "learned_c":
            c_vals = model.get_c_values()
            print(f"  c params: {len(c_vals)}")
            snapped = {"0.25": 0, "0.50": 0}
            for v in c_vals.values():
                if abs(v - 0.25) < 0.01:
                    snapped["0.25"] += 1
                elif abs(v - 0.50) < 0.01:
                    snapped["0.50"] += 1
            total = sum(snapped.values())
            if total > 0:
                print(
                    f"  snapped → 0.25: {snapped['0.25']}/{total}  "
                    f"0.50: {snapped['0.50']}/{total}"
                )
        print()

    # Summary table
    print("=" * 60)
    print(f"{'Baseline':<20s} {'Loss':>8s} {'PPL':>8s} {'Δ vs FP':>8s}")
    print("-" * 60)
    fp_ppl = results.get("full_precision", {}).get("ppl")
    for name in args.baselines:
        if name not in results:
            continue
        r = results[name]
        delta = f"{r['ppl'] - fp_ppl:+.2f}" if fp_ppl else "—"
        print(f"{name:<20s} {r['loss']:>8.4f} {r['ppl']:>8.2f} {delta:>8s}")
    print("=" * 60)


if __name__ == "__main__":
    main()
