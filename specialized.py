"""
Specialized inference backends for TetraNet baselines.

Each baseline runs its optimal arithmetic path:
  full_precision — FP32 BLAS matmul (F.linear)
  bitnet         — ternary MUX/add (no MULT)
  quaternary     — 2-bit packed shift matmul (no MULT)
  uniform_2bit   — 2-bit packed table / small MUL (not power-of-two)

Layout contract:
  nn.Linear weight is [out_features, in_features]
  F.linear(x, W) == x @ W.T
  Kernels compute A[M, K] @ B[K, N] with K=in_features, N=out_features
  so we pack weight.T (contiguous) for GEMM kernels.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quaternary import quaternary_weight_quantize

# ── optional native / triton backends ──────────────────────────────────────
_CPU_EXT = None
try:
    import sys
    from pathlib import Path

    _ext_dir = Path(__file__).resolve().parent / "csr_ext"
    if str(_ext_dir) not in sys.path:
        sys.path.insert(0, str(_ext_dir))
    import quant_matmul as _CPU_EXT  # type: ignore
except Exception:
    _CPU_EXT = None

_TRITON_SHIFT = None
_TRITON_TERN = None
_TRITON_UNI = None
try:
    from triton_kernels.shift_matmul import shift_matmul as _TRITON_SHIFT
    from triton_kernels.ternary_matmul import ternary_matmul as _TRITON_TERN
    from triton_kernels.uniform_matmul import uniform_matmul as _TRITON_UNI
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  Quantize (match training) + pack
# ═══════════════════════════════════════════════════════════════════════════


def bitnet_quantize_weight(weight: torch.Tensor, threshold: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (ternary in {-1,0,1}, gamma) matching BitNetTernaryLinear."""
    gamma = weight.abs().mean().clamp(min=1e-8).detach()
    scaled = weight / gamma
    q = torch.sign(scaled)
    q[scaled.abs() < 0.5] = 0.0
    return q, gamma


def uniform_quantize_weight(weight: torch.Tensor, threshold: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (levels in {-1,-1/3,1/3,1}, gamma) matching Uniform2BitLinear."""
    gamma = weight.abs().max().clamp(min=1e-8).detach()
    scaled = weight / gamma
    levels = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=weight.device, dtype=weight.dtype)
    flat = scaled.reshape(-1, 1)
    idx = (flat - levels.unsqueeze(0)).abs().argmin(dim=1)
    q = levels[idx].view_as(scaled)
    return q, gamma


def quaternary_quantize_weight(weight: torch.Tensor, c: float, threshold: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (values in {-1,-c,c,1}, gamma)."""
    c_t = torch.tensor(c, device=weight.device, dtype=weight.dtype)
    gamma = weight.abs().mean().clamp(min=1e-8).detach()
    q = quaternary_weight_quantize(weight, c_t, threshold) / gamma
    return q, gamma


def shift_bits_for_c(c: float) -> int:
    """Map snapped c to power-of-two shift amount."""
    if abs(c - 0.25) <= abs(c - 0.5):
        return 2
    return 1


def pack_ternary_int8(w_ternary: torch.Tensor) -> torch.Tensor:
    """Pack {-1,0,1} float/int tensor to int8 [K, N] (GEMM layout = weight.T)."""
    w = w_ternary.detach()
    out = torch.zeros_like(w, dtype=torch.int8)
    out[w > 0.5] = 1
    out[w < -0.5] = -1
    return out.contiguous()


def pack_quaternary_2bit(w_quat: torch.Tensor, c: float) -> torch.Tensor:
    """
    Pack quaternary values {-1,-c,c,1} along dim 0 into uint8.
    Input shape [K, N] (GEMM layout). Output [ceil(K/4), N].

    Codes: 00=+1, 01=-1, 10=+c, 11=-c
    """
    if _CPU_EXT is not None and w_quat.device.type == "cpu" and w_quat.dtype == torch.float32:
        return _CPU_EXT.pack_quaternary_weights(w_quat.contiguous(), float(c))

    boundary = (1.0 + c) / 2.0
    K, N = w_quat.shape
    K_packed = (K + 3) // 4
    packed = torch.zeros(K_packed, N, dtype=torch.uint8, device=w_quat.device)
    v = w_quat
    for k in range(K):
        pk = k // 4
        bit = (k % 4) * 2
        col = torch.zeros(N, dtype=torch.uint8, device=w_quat.device)
        row = v[k]
        col[row <= -boundary] = 1
        col[(row > -boundary) & (row <= 0)] = 3
        col[(row > 0) & (row <= boundary)] = 2
        col[row > boundary] = 0
        packed[pk] |= col << bit
    return packed


def pack_uniform_2bit(w_scaled: torch.Tensor) -> torch.Tensor:
    """
    Pack uniform levels {-1,-1/3,1/3,1} along dim 0 into uint8 [ceil(K/4), N].
    Codes: 0=-1, 1=-1/3, 2=+1/3, 3=+1
    """
    levels = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=w_scaled.device, dtype=w_scaled.dtype)
    K, N = w_scaled.shape
    K_packed = (K + 3) // 4
    packed = torch.zeros(K_packed, N, dtype=torch.uint8, device=w_scaled.device)
    for k in range(K):
        pk = k // 4
        bit = (k % 4) * 2
        row = w_scaled[k]
        dist = (row.unsqueeze(-1) - levels).abs()
        codes = dist.argmin(dim=-1).to(torch.uint8)
        packed[pk] |= codes << bit
    return packed


UNIFORM_LEVELS = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Activation quant
# ═══════════════════════════════════════════════════════════════════════════


def quantize_activation_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor absmax INT8 quant. Returns (int32 codes, scale)."""
    scale = x.detach().abs().amax().clamp(min=1e-8) / 127.0
    a = (x / scale).round().clamp(-127, 127).to(torch.int32)
    return a, scale


# ═══════════════════════════════════════════════════════════════════════════
#  Kernels (CPU ext / Triton / torch fallback)
# ═══════════════════════════════════════════════════════════════════════════


def _torch_ternary_matmul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """a [M,K] int32, w [K,N] int8 → [M,N] int32 via add/sub only."""
    M, K = a.shape
    _, N = w.shape
    acc = torch.zeros(M, N, dtype=torch.int32, device=a.device)
    w32 = w.to(torch.int32)
    for k in range(K):
        ak = a[:, k].unsqueeze(1)  # [M,1]
        wk = w32[k].unsqueeze(0)  # [1,N]
        acc += torch.where(wk == 1, ak, torch.zeros_like(ak))
        acc -= torch.where(wk == -1, ak, torch.zeros_like(ak))
    return acc


def _torch_shift_matmul(a: torch.Tensor, packed: torch.Tensor, shift_bits: int, K: int) -> torch.Tensor:
    """a [M,K] int32, packed [K_packed,N] uint8 → [M,N] int32."""
    M = a.shape[0]
    K_packed, N = packed.shape
    acc = torch.zeros(M, N, dtype=torch.int32, device=a.device)
    for pk in range(K_packed):
        byte = packed[pk].to(torch.int32)  # [N]
        for b in range(4):
            k = pk * 4 + b
            if k >= K:
                break
            code = (byte >> (b * 2)) & 3
            ak = a[:, k].unsqueeze(1)  # [M,1]
            shifted = ak >> shift_bits
            # code 0:+1, 1:-1, 2:+c, 3:-c
            term = torch.where(code == 0, ak, torch.zeros_like(ak))
            term = term - torch.where(code == 1, ak, torch.zeros_like(ak))
            term = term + torch.where(code == 2, shifted, torch.zeros_like(ak))
            term = term - torch.where(code == 3, shifted, torch.zeros_like(ak))
            acc += term
    return acc


def _torch_uniform_matmul(a: torch.Tensor, packed: torch.Tensor, K: int) -> torch.Tensor:
    """a [M,K] int32, packed uint8 → float32 [M,N] with levels {-1,-1/3,1/3,1}."""
    M = a.shape[0]
    K_packed, N = packed.shape
    levels = torch.tensor(UNIFORM_LEVELS, device=a.device, dtype=torch.float32)
    acc = torch.zeros(M, N, dtype=torch.float32, device=a.device)
    a_f = a.to(torch.float32)
    for pk in range(K_packed):
        byte = packed[pk].to(torch.int32)
        for b in range(4):
            k = pk * 4 + b
            if k >= K:
                break
            code = (byte >> (b * 2)) & 3
            ak = a_f[:, k].unsqueeze(1)
            # gather level per column
            lvl = levels[code]  # [N]
            acc += ak * lvl.unsqueeze(0)
    return acc


def ternary_matmul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """a [M,K] int32, w [K,N] int8 → int32 [M,N]."""
    if a.device.type == "cuda" and _TRITON_TERN is not None:
        return _TRITON_TERN(a.contiguous(), w.contiguous())
    if _CPU_EXT is not None and a.device.type == "cpu":
        return _CPU_EXT.ternary_matmul(a.contiguous(), w.contiguous())
    return _torch_ternary_matmul(a, w)


def shift_matmul(a: torch.Tensor, packed: torch.Tensor, shift_bits: int, K: int) -> torch.Tensor:
    """a [M,K] int32, packed [ceil(K/4),N] uint8 → int32 [M,N]."""
    if a.device.type == "cuda" and _TRITON_SHIFT is not None:
        return _TRITON_SHIFT(a.contiguous(), packed.contiguous(), shift_bits)
    if _CPU_EXT is not None and a.device.type == "cpu":
        return _CPU_EXT.shift_matmul(a.contiguous(), packed.contiguous(), int(shift_bits))
    return _torch_shift_matmul(a, packed, shift_bits, K)


def uniform_matmul(a: torch.Tensor, packed: torch.Tensor, K: int) -> torch.Tensor:
    """a [M,K] int32, packed → float32 [M,N]."""
    if a.device.type == "cuda" and _TRITON_UNI is not None:
        return _TRITON_UNI(a.contiguous(), packed.contiguous(), K)
    if _CPU_EXT is not None and a.device.type == "cpu" and hasattr(_CPU_EXT, "uniform_matmul"):
        return _CPU_EXT.uniform_matmul(a.contiguous(), packed.contiguous(), int(K))
    return _torch_uniform_matmul(a, packed, K)


def int32_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _CPU_EXT is not None and a.device.type == "cpu":
        return _CPU_EXT.int32_matmul(a.contiguous(), b.contiguous())
    return (a.to(torch.int64) @ b.to(torch.int64)).to(torch.int32)


def backend_info() -> dict:
    return {
        "cpu_ext": _CPU_EXT is not None,
        "triton_shift": _TRITON_SHIFT is not None,
        "triton_ternary": _TRITON_TERN is not None,
        "triton_uniform": _TRITON_UNI is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Specialized linear modules
# ═══════════════════════════════════════════════════════════════════════════


class SpecializedFP32Linear(nn.Module):
    """Optimal FP32 path: BLAS F.linear."""

    backend_name = "fp32_blas"

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("weight", weight.detach().contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.detach().contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class SpecializedTernaryLinear(nn.Module):
    """BitNet path: INT8 acts × ternary codes, MUX/add only."""

    backend_name = "ternary_mux"

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, threshold: float = 1.0):
        super().__init__()
        q, gamma = bitnet_quantize_weight(weight, threshold)
        # GEMM layout: [K, N] = weight.T
        w_t = q.T.contiguous()
        self.register_buffer("codes", pack_ternary_int8(w_t))
        self.register_buffer("gamma", gamma.reshape(()).float())
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        if bias is not None:
            self.register_buffer("bias", bias.detach().contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        a, act_scale = quantize_activation_int8(x2)
        y = ternary_matmul(a, self.codes).to(dtype=torch.float32)
        y = y * act_scale * self.gamma
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], self.out_features).to(dtype=x.dtype)


class SpecializedShiftLinear(nn.Module):
    """Quaternary path: INT8 acts × 2-bit packed shift matmul."""

    backend_name = "quaternary_shift"

    def __init__(self, weight: torch.Tensor, c: float, bias: Optional[torch.Tensor] = None, threshold: float = 1.0):
        super().__init__()
        # Snap c to nearest well for shift encoding
        c_snap = 0.25 if abs(c - 0.25) <= abs(c - 0.5) else 0.5
        q, gamma = quaternary_quantize_weight(weight, c_snap, threshold)
        w_t = q.T.contiguous().float()
        self.register_buffer("packed", pack_quaternary_2bit(w_t, c_snap))
        self.register_buffer("gamma", gamma.reshape(()).float())
        self.shift_bits = shift_bits_for_c(c_snap)
        self.c_snap = c_snap
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        if bias is not None:
            self.register_buffer("bias", bias.detach().contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        a, act_scale = quantize_activation_int8(x2)
        y = shift_matmul(a, self.packed, self.shift_bits, self.in_features).to(dtype=torch.float32)
        y = y * act_scale * self.gamma
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], self.out_features).to(dtype=x.dtype)


class SpecializedUniformLinear(nn.Module):
    """Uniform 2-bit path: INT8 acts × packed codes with level table (uses MUL)."""

    backend_name = "uniform_2bit_lut"

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, threshold: float = 1.0):
        super().__init__()
        q, gamma = uniform_quantize_weight(weight, threshold)
        w_t = q.T.contiguous().float()
        self.register_buffer("packed", pack_uniform_2bit(w_t))
        self.register_buffer("gamma", gamma.reshape(()).float())
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        if bias is not None:
            self.register_buffer("bias", bias.detach().contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        a, act_scale = quantize_activation_int8(x2)
        y = uniform_matmul(a, self.packed, self.in_features)
        y = y * act_scale * self.gamma
        if self.bias is not None:
            y = y + self.bias
        return y.view(*shape[:-1], self.out_features).to(dtype=x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
#  Model specialization
# ═══════════════════════════════════════════════════════════════════════════


def _is_quant_linear(module: nn.Module) -> bool:
    return hasattr(module, "weight") and (
        hasattr(module, "_ternary_quantize")
        or hasattr(module, "_uniform_quantize")
        or hasattr(module, "c")
    )


def _baseline_kind(module: nn.Module) -> str:
    if hasattr(module, "_ternary_quantize"):
        return "bitnet"
    if hasattr(module, "_uniform_quantize"):
        return "uniform_2bit"
    if hasattr(module, "c"):
        return "quaternary"
    return "full_precision"


def specialize_linear(module: nn.Module, force: Optional[str] = None) -> nn.Module:
    """Convert one linear module into its specialized inference counterpart."""
    kind = force or _baseline_kind(module)
    w = module.weight.detach()
    b = module.bias.detach() if getattr(module, "bias", None) is not None else None
    thr = float(getattr(module, "threshold", 1.0))

    if kind == "full_precision":
        return SpecializedFP32Linear(w, b)
    if kind == "bitnet":
        return SpecializedTernaryLinear(w, b, thr)
    if kind == "uniform_2bit":
        return SpecializedUniformLinear(w, b, thr)
    if kind == "quaternary":
        c = module.c.item() if isinstance(module.c, torch.Tensor) else float(module.c)
        return SpecializedShiftLinear(w, c, b, thr)
    raise ValueError(f"unknown kind {kind}")


_PROJ_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def _force_kind(baseline: Optional[str]) -> Optional[str]:
    if baseline is None:
        return None
    if baseline == "full_precision":
        return "full_precision"
    if baseline == "bitnet":
        return "bitnet"
    if baseline == "uniform_2bit":
        return "uniform_2bit"
    if baseline.startswith("fixed_c") or baseline in ("learned_c", "heterogeneous", "quaternary"):
        return "quaternary"
    return baseline


def specialize_model(model: nn.Module, baseline: Optional[str] = None) -> nn.Module:
    """
    Replace projection linears with specialized inference backends.

    baseline:
      None — auto-detect per module
      full_precision | bitnet | uniform_2bit | quaternary | fixed_c_* | learned_c | heterogeneous
    """
    force = _force_kind(baseline)
    to_replace = []

    for name, module in model.named_modules():
        if name == "" or "." not in name:
            continue
        child_name = name.rsplit(".", 1)[-1]
        parent_name = name.rsplit(".", 1)[0]
        if child_name not in _PROJ_NAMES:
            continue
        if isinstance(
            module,
            (
                SpecializedFP32Linear,
                SpecializedTernaryLinear,
                SpecializedShiftLinear,
                SpecializedUniformLinear,
            ),
        ):
            continue
        if not isinstance(module, nn.Linear):
            continue
        to_replace.append((parent_name, child_name, module))

    for parent_name, child_name, module in to_replace:
        parent = model.get_submodule(parent_name)
        kind = force if force is not None else _baseline_kind(module)
        setattr(parent, child_name, specialize_linear(module, force=kind))

    model._specialized = True
    model._specialized_baseline = baseline
    return model


def infer_config_from_state_dict(sd: dict) -> dict:
    """Infer model config from checkpoint state dict (same approach as eval.py)."""
    vocab_size, hidden_dim = sd["embed_tokens.weight"].shape
    layer_keys = sorted(
        {k.split(".")[1] for k in sd if k.startswith("layers.") and k.split(".")[1].isdigit()},
        key=int,
    )
    num_layers = len(layer_keys)
    # ffn from gate_proj
    ffn_dim = None
    for k, v in sd.items():
        if "gate_proj.weight" in k:
            ffn_dim = v.shape[0]
            break
    if ffn_dim is None:
        ffn_dim = 4 * hidden_dim
    # heads: assume head_dim 32 for 256d/8h or 64 for 768d/12h
    if hidden_dim % 64 == 0 and hidden_dim >= 768:
        num_heads = hidden_dim // 64
    elif hidden_dim % 32 == 0:
        num_heads = hidden_dim // 32
    else:
        num_heads = max(1, hidden_dim // 64)
    return dict(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=512,
        tie_weights=True,
    )


def compressed_size_mb(model: nn.Module) -> float:
    """Estimate packed weight storage in MB."""
    total_bytes = 0
    for m in model.modules():
        if isinstance(m, SpecializedFP32Linear):
            total_bytes += m.weight.numel() * 4
        elif isinstance(m, SpecializedTernaryLinear):
            # ~1.58 bit ideal → store as 2-bit: 4 weights/byte + 4B gamma
            total_bytes += math.ceil(m.codes.numel() / 4) + 4
        elif isinstance(m, SpecializedShiftLinear):
            total_bytes += m.packed.numel() + 4
        elif isinstance(m, SpecializedUniformLinear):
            total_bytes += m.packed.numel() + 4
    # embeddings + norms roughly FP32
    for name, p in model.named_parameters():
        if "embed" in name or "norm" in name or "lm_head" in name:
            total_bytes += p.numel() * 4
    for name, b in model.named_buffers():
        if "embed" in name:
            total_bytes += b.numel() * 4
    if total_bytes == 0:
        total_bytes = sum(p.numel() * 4 for p in model.parameters())
    return total_bytes / (1024**2)
