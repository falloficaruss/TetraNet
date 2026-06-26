"""Train a custom Byte-Level BPE tokenizer (vocab_size=4096) on TinyStories."""

import argparse

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from datasets import load_dataset


def train_tokenizer(vocab_size: int, output_path: str, num_stories: int | None = None):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    def get_training_corpus():
        for i, example in enumerate(dataset):
            if num_stories is not None and i >= num_stories:
                break
            yield example["text"]

    tokenizer.train_from_iterator(get_training_corpus(), trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)],
    )

    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path} (vocab_size={tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--output", default="./tetranet_tokenizer.json")
    parser.add_argument("--num-stories", type=int, default=None,
                        help="Limit training stories (default: all 2.1M)")
    args = parser.parse_args()
    train_tokenizer(args.vocab_size, args.output, args.num_stories)
