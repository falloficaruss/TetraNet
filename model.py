"""
Llama-style Decoder-only Transformer using QBitLinearQuaternary layers.
~9.4M parameters at default config (vocab_size=4096). Designed for TinyStories training.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quaternary import QBitLinearQuaternary


@dataclass
class QuaternaryLlamaConfig:
    """Hyperparameters for the Quaternary Llama model (~9.4M params at defaults)."""

    vocab_size: int = 4096
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 1024  # 4x hidden_dim for SwiGLU
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    initial_c: float = 0.375
    threshold: float = 1.0
    rms_norm_eps: float = 1e-6
    tie_weights: bool = True
    linear_cls: type = QBitLinearQuaternary

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


class RMSNorm(nn.Module):
    """RMS Layer Normalization (used in Llama)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings."""

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim // 2)

        self.register_buffer(
            "cos", freqs.cos().unsqueeze(0).unsqueeze(1), persistent=False
        )  # (1, 1, max_seq_len, dim//2)
        self.register_buffer(
            "sin", freqs.sin().unsqueeze(0).unsqueeze(1), persistent=False
        )

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:, :, :seq_len], self.sin[:, :, :seq_len]


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    Args:
        xq, xk: (batch, num_heads, seq_len, head_dim)
        cos, sin: (1, 1, seq_len, head_dim//2) — broadcast over batch & heads
    """
    head_dim = xq.shape[-1]
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Split into two halves and apply rotation
    xq_half = xq.reshape(*xq.shape[:-1], -1, 2).unbind(-1)  # (B, H, T, d//2) each
    xk_half = xk.reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    xq_rotated = torch.stack(
        [xq_half[0] * cos - xq_half[1] * sin, xq_half[0] * sin + xq_half[1] * cos],
        dim=-1,
    ).flatten(-2)

    xk_rotated = torch.stack(
        [xk_half[0] * cos - xk_half[1] * sin, xk_half[0] * sin + xk_half[1] * cos],
        dim=-1,
    ).flatten(-2)

    return xq_rotated, xk_rotated


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP with quantized projections."""

    def __init__(self, config: QuaternaryLlamaConfig):
        super().__init__()
        self.gate_proj = config.linear_cls(
            config.hidden_dim,
            config.ffn_dim,
            bias=False,
            initial_c=config.initial_c,
            threshold=config.threshold,
        )
        self.up_proj = config.linear_cls(
            config.hidden_dim,
            config.ffn_dim,
            bias=False,
            initial_c=config.initial_c,
            threshold=config.threshold,
        )
        self.down_proj = config.linear_cls(
            config.ffn_dim,
            config.hidden_dim,
            bias=False,
            initial_c=config.initial_c,
            threshold=config.threshold,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with quantized projections and RoPE."""

    def __init__(self, config: QuaternaryLlamaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        self.q_proj = config.linear_cls(
            config.hidden_dim,
            config.hidden_dim,
            bias=False,
            initial_c=config.initial_c,
            threshold=config.threshold,
        )
        self.k_proj = config.linear_cls(
            config.hidden_dim,
            config.hidden_dim,
            bias=False,
            initial_c=config.initial_c,
            threshold=config.threshold,
        )
        self.v_proj = config.linear_cls(
            config.hidden_dim,
            config.hidden_dim,
            bias=False,
            initial_c=config.initial_c,
            threshold=config.threshold,
        )
        self.o_proj = config.linear_cls(
            config.hidden_dim,
            config.hidden_dim,
            bias=False,
            initial_c=config.initial_c,
            threshold=config.threshold,
        )

        self.rotary = RotaryEmbedding(
            self.head_dim, config.max_seq_len, config.rope_base
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V and reshape for multi-head attention
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary(T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Scaled dot-product attention with causal masking
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        y = attn @ v  # (B, num_heads, T, head_dim)

        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class DecoderLayer(nn.Module):
    """One transformer decoder layer with pre-norm: attn → residual → mlp → residual."""

    def __init__(self, config: QuaternaryLlamaConfig):
        super().__init__()
        self.self_attn = CausalSelfAttention(config)
        self.mlp = SwiGLUMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, causal_mask)
        x = residual + x

        # Pre-norm MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class QuaternaryLlamaForCausalLM(nn.Module):
    """Llama-style decoder-only transformer with quaternary weight quantization.

    Default: ~11M parameters, 8 layers, 256 hidden dim, 8 heads.
    """

    def __init__(self, config: Optional[QuaternaryLlamaConfig] = None):
        super().__init__()
        self.config = config or QuaternaryLlamaConfig()

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(self.config) for _ in range(self.config.num_layers)]
        )
        self.norm = RMSNorm(self.config.hidden_dim, self.config.rms_norm_eps)
        self.lm_head = nn.Linear(
            self.config.hidden_dim, self.config.vocab_size, bias=False
        )

        # Precompute causal mask as a buffer
        mask = torch.triu(
            torch.ones(
                self.config.max_seq_len, self.config.max_seq_len, dtype=torch.bool
            ),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

        # Weight tying
        if self.config.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution (standard transformer init)."""
        std = 0.02
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim >= 2:
                nn.init.normal_(param, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len) token indices.
            labels: Optional (batch_size, seq_len) target token indices for computing loss.

        Returns:
            dict with 'logits' and optionally 'loss'.
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, (
            f"Input sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Token embeddings
        x = self.embed_tokens(input_ids)  # (B, T, hidden_dim)

        # Apply transformer layers
        causal_mask = self.causal_mask[:T, :T]  # (T, T) — broadcast to (B, H, T, T)
        for layer in self.layers:
            x = layer(x, causal_mask)

        # Final normalization
        x = self.norm(x)

        # Language modeling head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        outputs = {"logits": logits}

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            outputs["loss"] = loss

        return outputs

    def get_num_params(self) -> int:
        """Returns total number of parameters, handling tied weights correctly."""
        return sum(p.numel() for p in set(self.parameters()))

    def get_c_values(self) -> dict[str, float]:
        """Returns a dict mapping projection -> c value for all layers.
        Empty dict for baselines without learnable c parameters."""
        c_vals = {}
        for i, layer in enumerate(self.layers):
            for name, param in layer.named_parameters():
                if name.endswith(".c"):
                    c_vals[f"layer.{i}.{name.replace('.c', '')}"] = param.item()
        return c_vals
