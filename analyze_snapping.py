import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from models import build_model

SEQ_LEN = 512
BATCH_SIZE = 4
MAX_STORIES = 500

WELL_CONFIGS = {
    "A [0.25, 0.5]": [0.25, 0.5],
    "B [0.25, 0.5, 0.75, 1.0]": [0.25, 0.5, 0.75, 1.0],
    "C [0, 0.25, 0.75, 1.0]": [0, 0.25, 0.75, 1.0],
}


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


def infer_config_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    vocab_size = sd["embed_tokens.weight"].shape[0]
    hidden_dim = sd["embed_tokens.weight"].shape[1]
    layer_keys = set()
    for k in sd:
        if k.startswith("layers."):
            layer_keys.add(int(k.split(".")[1]))
    num_layers = len(layer_keys)
    ffn_dim = sd["layers.0.mlp.down_proj.weight"].shape[1]
    num_heads = 12
    return {
        "vocab_size": vocab_size,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "max_seq_len": SEQ_LEN,
        "tie_weights": True,
    }, ckpt


def collect_c_params(model):
    params = []
    for name, module in model.named_modules():
        if hasattr(module, "c") and isinstance(module.c, torch.nn.Parameter):
            params.append((name, module.c))
    return params


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


def snap_c_values(c_params, wells):
    wells_t = torch.tensor(wells)
    for name, c_param in c_params:
        dist = (c_param.data - wells_t).abs()
        nearest = wells_t[dist.argmin()]
        c_param.data.copy_(nearest)


def main():
    parser = argparse.ArgumentParser(
        description="Post-training snapping analysis for quaternary model"
    )
    parser.add_argument(
        "ckpt_path",
        nargs="?",
        default="checkpoint_final_quaternary.pt",
        help="Path to quaternary checkpoint (no-snap version)",
    )
    parser.add_argument(
        "--tokenizer", default="tetranet_tokenizer.json"
    )
    parser.add_argument(
        "--valid-data", default="tinystories/TinyStoriesV2-GPT4-valid.txt"
    )
    parser.add_argument(
        "--max-stories", type=int, default=MAX_STORIES
    )
    args = parser.parse_args()

    cfg, ckpt = infer_config_from_ckpt(args.ckpt_path)
    print(
        f"Model config: vocab={cfg['vocab_size']}, dim={cfg['hidden_dim']}, "
        f"layers={cfg['num_layers']}, heads={cfg['num_heads']}, ffn={cfg['ffn_dim']}"
    )

    tokenizer = load_tokenizer(args.tokenizer)
    tokens = tokenize_valid(args.valid_data, tokenizer, args.max_stories)
    dataset = ValidDataset(tokens, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"Validation sequences: {len(dataset):,}")

    print("Building model...", flush=True)
    model = build_model(
        baseline="quaternary",
        vocab_size=cfg["vocab_size"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        ffn_dim=cfg["ffn_dim"],
        max_seq_len=cfg["max_seq_len"],
        tie_weights=cfg["tie_weights"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    c_params = collect_c_params(model)
    original_c_vals = {name: p.data.clone() for name, p in c_params}
    print(f"\nFound {len(c_params)} c parameters\n")

    # Print original c distribution
    print("Original c values:")
    print(f"  {'Module':<40s} {'c':>8s}  {'Nearest well'}")
    print(f"  {'-'*40} {'-'*8}  {'-'*12}")
    for name, c_param in c_params:
        cv = c_param.item()
        nearest = min([0.25, 0.5, 0.75, 1.0], key=lambda w: abs(cv - w))
        print(f"  {name:<40s} {cv:>8.4f}  {nearest}")
    print()

    # Evaluate original PPL
    print("Evaluating original (no snap) PPL...", flush=True)
    loss_orig, ppl_orig = eval_ppl(model, loader)
    print(f"  PPL original: {ppl_orig:.2f}  (loss: {loss_orig:.4f})\n")

    # Test each well configuration
    results = {}
    for config_name, wells in WELL_CONFIGS.items():
        print(f"Testing {config_name}...", flush=True)
        snap_c_values(c_params, wells)

        loss_snap, ppl_snap = eval_ppl(model, loader)
        results[config_name] = (loss_snap, ppl_snap)
        print(f"  PPL {config_name}: {ppl_snap:.2f}  (loss: {loss_snap:.4f})  "
              f"Δ: {ppl_snap - ppl_orig:+.2f}\n")

        # Restore original c
        for name, c_param in c_params:
            c_param.data.copy_(original_c_vals[name])

    # Summary table
    print("=" * 60)
    print(f"{'Configuration':<30s} {'PPL':>8s} {'Δ':>8s}")
    print("-" * 60)
    print(f"{'Original (no snap)':<30s} {ppl_orig:>8.2f} {'—':>8s}")
    for config_name, (loss_snap, ppl_snap) in results.items():
        delta = ppl_snap - ppl_orig
        print(f"{config_name:<30s} {ppl_snap:>8.2f} {delta:>+8.2f}")
    print("=" * 60)

    # Distance analysis
    print(f"\n{'Distance to nearest well':^60s}")
    print(f"  {'Module':<40s} {'c (orig)':>10s} {'Nearest well':>12s} {'Distance':>10s}")
    print(f"  {'-'*40} {'-'*10} {'-'*12} {'-'*10}")
    for name, c_param in c_params:
        cv = original_c_vals[name].item()
        nearest, dist = min(
            [(w, abs(cv - w)) for w in [0.25, 0.5, 0.75, 1.0]], key=lambda x: x[1]
        )
        marker = " ✓" if dist < 0.02 else ""
        print(f"  {name:<40s} {cv:>10.4f} {nearest:>12.4f} {dist:>10.4f}{marker}")


if __name__ == "__main__":
    main()
