"""
Specialized inference engine for TetraNet baselines.

Each baseline uses its optimal arithmetic path:
  full_precision — FP32 BLAS
  bitnet         — ternary MUX/add
  quaternary*    — 2-bit shift matmul
  uniform_2bit   — 2-bit LUT/mul

Usage:
    model = load_and_optimize("checkpoint_fixed_c_05_final.pt", "fixed_c_05")
    tokens = generate(model, prompt_ids, max_new_tokens=100)
    stats = benchmark_generation(model, prompt_ids, max_new_tokens=200)
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import apply_rotary_emb
from specialized import (
    specialize_model,
    infer_config_from_state_dict,
    compressed_size_mb,
    backend_info,
    SpecializedFP32Linear,
    SpecializedTernaryLinear,
    SpecializedShiftLinear,
    SpecializedUniformLinear,
)


def _build_model(baseline: str, config: dict):
    from train_kaggle import build_model

    return build_model(baseline=baseline, **config)


# ═══════════════════════════════════════════════════════════════════════════
#  KV-cache patching
# ═══════════════════════════════════════════════════════════════════════════


def _patch_attention_kv(attn_module: nn.Module) -> None:
    orig_rotary = attn_module.rotary

    def _kv_forward(self, x, past_kv=None, use_cache=False, position=0):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            cos = orig_rotary.cos[:, :, position : position + T]
            sin = orig_rotary.sin[:, :, position : position + T]
            q, k = apply_rotary_emb(q, k, cos, sin)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        else:
            cos, sin = orig_rotary(T)
            q, k = apply_rotary_emb(q, k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        if past_kv is None:
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        output = self.o_proj(y)

        if use_cache or past_kv is not None:
            return output, (k, v)
        return output

    attn_module.forward = _kv_forward.__get__(attn_module, type(attn_module))
    attn_module._kv_enabled = True


def _patch_decoder_kv(decoder_module: nn.Module) -> None:
    def _kv_decoder_forward(self, x, past_kv=None, use_cache=False, position=0):
        residual = x
        x = self.input_layernorm(x)
        new_kv = None
        if past_kv is not None or use_cache:
            x, new_kv = self.self_attn(x, past_kv=past_kv, use_cache=use_cache, position=position)
        else:
            x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        if past_kv is not None or use_cache:
            return x, new_kv
        return x

    decoder_module.forward = _kv_decoder_forward.__get__(decoder_module, type(decoder_module))
    decoder_module._kv_enabled = True


def enable_kv_cache(model: nn.Module) -> nn.Module:
    for module in model.modules():
        name = type(module).__name__
        if name == "CausalSelfAttention" and not getattr(module, "_kv_enabled", False):
            _patch_attention_kv(module)
        elif name == "DecoderLayer" and not getattr(module, "_kv_enabled", False):
            _patch_decoder_kv(module)
    model._kv_enabled = True
    return model


def model_has_kv(model: nn.Module) -> bool:
    return bool(getattr(model, "_kv_enabled", False))


# ═══════════════════════════════════════════════════════════════════════════
#  Generation
# ═══════════════════════════════════════════════════════════════════════════


def _sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.clone()
        logits[logits < v[:, -1:]] = float("-inf")

    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.clone()
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def _generate_no_kv(
    model, input_ids, max_new_tokens, temperature, top_k, top_p, eos_token_id, max_seq_len
):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        if generated.shape[1] >= max_seq_len:
            break
        out = model(generated)
        logits = out["logits"]
        next_token = _sample(logits[:, -1, :], temperature, top_k, top_p)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        if eos_token_id is not None and (next_token == eos_token_id).any():
            break
    return generated


@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is not None:
        model = model.to(device)
        input_ids = input_ids.to(device)
    model.eval()

    max_seq_len = getattr(model.config, "max_seq_len", 2048)
    has_kv = model_has_kv(model)

    if not has_kv:
        return _generate_no_kv(
            model, input_ids, max_new_tokens, temperature, top_k, top_p, eos_token_id, max_seq_len
        )

    generated = input_ids.clone()
    past_kv = []
    x = model.embed_tokens(input_ids)
    for layer in model.layers:
        x, layer_kv = layer(x, use_cache=True, position=0)
        past_kv.append(layer_kv)
    x = model.norm(x)
    logits = model.lm_head(x)

    next_token = _sample(logits[:, -1, :], temperature, top_k, top_p)
    generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

    if eos_token_id is not None and (next_token == eos_token_id).any():
        return generated

    for _ in range(1, max_new_tokens):
        if generated.shape[1] >= max_seq_len:
            break
        x = model.embed_tokens(next_token.unsqueeze(-1))
        position = past_kv[0][0].shape[2]
        new_past_kv = []
        for i, layer in enumerate(model.layers):
            x, layer_kv = layer(x, past_kv=past_kv[i], use_cache=True, position=position)
            new_past_kv.append(layer_kv)
        past_kv = new_past_kv
        x = model.norm(x)
        logits = model.lm_head(x)
        next_token = _sample(logits[:, -1, :], temperature, top_k, top_p)
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
        if eos_token_id is not None and (next_token == eos_token_id).any():
            break

    return generated


# ═══════════════════════════════════════════════════════════════════════════
#  Benchmarking
# ═══════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def _do_prefill(model, input_ids):
    past_kv = []
    x = model.embed_tokens(input_ids)
    for layer in model.layers:
        x, layer_kv = layer(x, use_cache=True, position=0)
        past_kv.append(layer_kv)
    x = model.norm(x)
    return past_kv, x


@torch.no_grad()
def _do_generate_full(model, input_ids, max_new_tokens, temperature):
    past_kv, h = _do_prefill(model, input_ids)
    logits = model.lm_head(h)
    next_token = _sample(logits[:, -1, :], temperature)

    for _ in range(max_new_tokens - 1):
        x = model.embed_tokens(next_token.unsqueeze(-1))
        position = past_kv[0][0].shape[2]
        new_past_kv = []
        for i, layer in enumerate(model.layers):
            x, layer_kv = layer(x, past_kv=past_kv[i], use_cache=True, position=position)
            new_past_kv.append(layer_kv)
        past_kv = new_past_kv
        x = model.norm(x)
        logits = model.lm_head(x)
        next_token = _sample(logits[:, -1, :], temperature)


def _collect_backend_tags(model: nn.Module) -> str:
    tags = set()
    for m in model.modules():
        if isinstance(m, SpecializedFP32Linear):
            tags.add("fp32_blas")
        elif isinstance(m, SpecializedTernaryLinear):
            tags.add("ternary_mux")
        elif isinstance(m, SpecializedShiftLinear):
            tags.add("quaternary_shift")
        elif isinstance(m, SpecializedUniformLinear):
            tags.add("uniform_2bit_lut")
    return ",".join(sorted(tags)) if tags else "unspecialized"


@torch.no_grad()
def benchmark_generation(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 200,
    n_warmup: int = 2,
    n_runs: int = 5,
    temperature: float = 0.0,
    device: Optional[torch.device] = None,
) -> dict:
    if device is not None:
        model = model.to(device)
        prompt_ids = prompt_ids.to(device)
    model.eval()
    torch_device = next(model.parameters()).device if any(True for _ in model.parameters()) else prompt_ids.device
    # specialized modules use buffers
    try:
        torch_device = next(model.buffers()).device
    except StopIteration:
        pass

    B, prompt_len = prompt_ids.shape
    has_kv = model_has_kv(model)
    max_seq_len = getattr(model.config, "max_seq_len", 2048)

    def _sync():
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    # Prefill
    for _ in range(n_warmup):
        if has_kv:
            _do_prefill(model, prompt_ids)
        else:
            model(prompt_ids)

    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        if has_kv:
            _do_prefill(model, prompt_ids)
        else:
            model(prompt_ids)
    _sync()
    prefill_elapsed = time.perf_counter() - t0
    prefill_ms = prefill_elapsed / n_runs * 1000
    prefill_tok_s = B * prompt_len / (prefill_elapsed / n_runs)

    peak_mb = 0.0
    if torch_device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

    # Full generate timing (prefill+decode)
    for _ in range(n_warmup):
        if has_kv:
            _do_generate_full(model, prompt_ids, max_new_tokens, temperature)
        else:
            _generate_no_kv(
                model, prompt_ids, max_new_tokens, temperature, None, None, None, max_seq_len
            )

    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        if has_kv:
            _do_generate_full(model, prompt_ids, max_new_tokens, temperature)
        else:
            _generate_no_kv(
                model, prompt_ids, max_new_tokens, temperature, None, None, None, max_seq_len
            )
    _sync()
    total_elapsed = time.perf_counter() - t0
    total_ms = total_elapsed / n_runs * 1000
    decode_ms = max(total_ms - prefill_ms, 0.0)
    decode_tok_s = B * max_new_tokens / (decode_ms / 1000) if decode_ms > 0 else 0.0
    total_tok_s = B * (prompt_len + max_new_tokens) / (total_elapsed / n_runs)

    if torch_device.type == "cuda":
        peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / 1e6)

    return {
        "decode_tok_s": decode_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "total_tok_s": total_tok_s,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "peak_mb": peak_mb,
        "compressed_mb": compressed_size_mb(model),
        "has_kv": has_kv,
        "backend": _collect_backend_tags(model),
        "prompt_len": prompt_len,
        "max_new_tokens": max_new_tokens,
        "kernels": backend_info(),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Load + specialize
# ═══════════════════════════════════════════════════════════════════════════


def load_and_optimize(
    checkpoint_path: str,
    baseline: str,
    config: Optional[dict] = None,
    device: str = "cpu",
    enable_kv: bool = True,
) -> nn.Module:
    """Load checkpoint, specialize linears, optionally enable KV-cache."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if config is None:
        config = infer_config_from_state_dict(sd)

    baseline_cls = baseline.replace("_slow", "")
    model = _build_model(baseline_cls, config)
    model.load_state_dict(sd, strict=False)
    model.to(device)

    specialize_model(model, baseline=baseline_cls)
    if enable_kv:
        enable_kv_cache(model)

    model.eval()
    return model
