"""Train a custom Byte-Level BPE tokenizer (vocab_size=4096) on TinyStories text file."""

import argparse

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors


def train_tokenizer(vocab_size: int, data_path: str, output_path: str):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train([data_path], trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )

    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path} (vocab_size={tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--data-path", default="./tinystories/TinyStories-train.txt")
    parser.add_argument("--output", default="./tetranet_tokenizer.json")
    args = parser.parse_args()
    train_tokenizer(args.vocab_size, args.data_path, args.output)
