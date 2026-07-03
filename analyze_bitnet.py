import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from models import build_model

import models.layers as layers

SEQ_LEN = 512
BATCH_SIZE = 4
MAX_STORIES = 500


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


def collect_bitnet_modules(model):
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, layers.BitNetTernaryLinear):
            modules.append((name, module))
    return modules


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


def ternarize_weight(weight):
    gamma = weight.abs().mean().clamp(min=1e-8)
    scaled = weight / gamma
    quantized = torch.sign(scaled)
    quantized[scaled.abs() < 0.5] = 0.0
    return quantized * gamma


def main():
    parser = argparse.ArgumentParser(
        description="Post-training snapping analysis for BitNet"
    )
    parser.add_argument(
        "ckpt_path",
        nargs="?",
        default="checkpoint_final_bitnet.pt",
    )
    parser.add_argument("--tokenizer", default="tetranet_tokenizer.json")
    parser.add_argument(
        "--valid-data", default="tinystories/TinyStoriesV2-GPT4-valid.txt"
    )
    parser.add_argument("--max-stories", type=int, default=MAX_STORIES)
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
        baseline="bitnet",
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

    bitnet_modules = collect_bitnet_modules(model)
    print(f"Found {len(bitnet_modules)} BitNetTernaryLinear modules")

    # ── 1. Normal eval (on-the-fly quantization) ──
    print("\nEvaluating normal BitNet (on-the-fly quant)...", flush=True)
    loss_normal, ppl_normal = eval_ppl(model, loader)
    print(f"  PPL: {ppl_normal:.2f}  (loss: {loss_normal:.4f})")

    # ── 2. Eval with raw weights (no quantization) ──
    print("\nEvaluating raw weights (no quantization)...", flush=True)
    original_forwards = {}
    for name, module in bitnet_modules:
        original_forwards[name] = module.forward
        module.forward = lambda x, m=module: F.linear(x, m.weight, m.bias)
    loss_raw, ppl_raw = eval_ppl(model, loader)
    print(f"  PPL: {ppl_raw:.2f}  (loss: {loss_raw:.4f})  Δ: {ppl_raw - ppl_normal:+.2f}")
    # Restore forwards
    for name, module in bitnet_modules:
        module.forward = original_forwards[name]

    # ── 3. Post-snap: ternarize weights + plain forward ──
    print("\nEvaluating post-snapped weights (ternarize + plain forward)...", flush=True)
    original_weights = {}
    for name, module in bitnet_modules:
        original_weights[name] = module.weight.data.clone()
        module.weight.data.copy_(ternarize_weight(module.weight.data))
        module.forward = lambda x, m=module: F.linear(x, m.weight, m.bias)
    loss_snap, ppl_snap = eval_ppl(model, loader)
    print(f"  PPL: {ppl_snap:.2f}  (loss: {loss_snap:.4f})  Δ: {ppl_snap - ppl_normal:+.2f}")
    # Restore
    for name, module in bitnet_modules:
        module.weight.data.copy_(original_weights[name])
        module.forward = original_forwards[name]

    # ── 4. Weight statistics ──
    print(f"\n{'Weight statistics':^60s}")
    print(f"  {'Module':<45s} {'|raw - quantized| mean':>20s} {'unique vals':>12s}")
    print(f"  {'-'*45} {'-'*20} {'-'*12}")
    for name, module in bitnet_modules:
        w = module.weight.data
        qw = ternarize_weight(w)
        diff = (w - qw).abs().mean().item()
        uniq = qw.unique().tolist()
        print(f"  {name:<45s} {diff:>20.4f} {len(uniq):>5d}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"{'Mode':<30s} {'PPL':>8s} {'Δ':>8s}")
    print(f"{'-'*46}")
    print(f"{'Normal (on-the-fly ternary)':<30s} {ppl_normal:>8.2f} {'—':>8s}")
    print(f"{'Raw weights (no quant)':<30s} {ppl_raw:>8.2f} {ppl_raw - ppl_normal:>+8.2f}")
    print(f"{'Post-snapped weights':<30s} {ppl_snap:>8.2f} {ppl_snap - ppl_normal:>+8.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
