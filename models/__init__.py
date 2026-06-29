"""
Model factory — builds any of the 4 baselines by swapping the linear layer class.

Usage:
    from models import build_model, BASELINES
    model = build_model(baseline="full_precision", vocab_size=4096, ...)
    model = build_model(baseline="bitnet", ...)
    model = build_model(baseline="uniform_2bit", ...)
    model = build_model(baseline="quaternary", ...)
"""

from model import QuaternaryLlamaForCausalLM, QuaternaryLlamaConfig
from models.layers import FullPrecisionLinear, BitNetTernaryLinear, Uniform2BitLinear
from quaternary import QBitLinearQuaternary

BASELINES = {
    "full_precision": FullPrecisionLinear,
    "bitnet": BitNetTernaryLinear,
    "uniform_2bit": Uniform2BitLinear,
    "quaternary": QBitLinearQuaternary,
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
        raise ValueError(f"Unknown baseline: {baseline}. Choose from {list(BASELINES.keys())}")

    linear_cls = BASELINES[baseline]

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
        linear_cls=linear_cls,
    )

    return QuaternaryLlamaForCausalLM(config)
