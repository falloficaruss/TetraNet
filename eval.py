from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from models import build_model

BASELINE = "quaternary"
CKPT_PATH = "checkpoint_final_quaternary.pt"
TOKENIZER_PATH = "tetranet_tokenizer.json"
VALID_PATH = "tinystories/TinyStoriesV2-GPT4-valid.txt"
SEQ_LEN = 512
BATCH_SIZE = 4
MAX_STORIES = 2500  # -1 for all stories (~6M tokens, ~7h on CPU)


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


def main():
    cfg, ckpt = infer_config_from_ckpt(CKPT_PATH)
    print(
        f"Model config: vocab={cfg['vocab_size']}, dim={cfg['hidden_dim']}, "
        f"layers={cfg['num_layers']}, heads={cfg['num_heads']}, ffn={cfg['ffn_dim']}"
    )

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    tokens = tokenize_valid(VALID_PATH, tokenizer, MAX_STORIES)

    dataset = ValidDataset(tokens, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"Validation sequences: {len(dataset):,}")

    print("Building model...", flush=True)
    model = build_model(
        baseline=BASELINE,
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
    print("Model ready. Starting eval loop...", flush=True)

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            out = model(input_ids, labels=labels)
            loss = out["loss"]
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1
            if num_batches % 500 == 0:
                print(
                    f"  processed {num_batches}/{len(loader)} batches ({total_tokens:,} tokens)",
                    flush=True,
                )

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"\n{'=' * 40}")
    print(f"Baseline: {BASELINE}")
    print(f"Validation loss: {avg_loss:.4f}")
    print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
