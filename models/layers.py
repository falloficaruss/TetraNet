"""
Quantized linear layer implementations for all 4 baselines:

1. FullPrecisionLinear   — no quantization (theoretical upper bound)
2. BitNetTernaryLinear   — BitNet b1.58 ternary {-1, 0, 1} with STE
3. Uniform2BitLinear     — Uniform 2-bit 4-level quant with STE
4. QuaternaryLinear      — Proposed power-of-two quaternary {-1, -c, c, 1} (re-export)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export for convenience
from quaternary import FixedCQuaternaryLinear as QuaternaryLinear


class FullPrecisionLinear(nn.Linear):
    """Standard nn.Linear — no quantization.
    Serves as the theoretical upper bound baseline."""

    def __init__(self, in_features, out_features, bias=False, **kwargs):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class BitNetTernaryFunction(torch.autograd.Function):
    """STE backward for BitNet b1.58 ternary quantization."""

    @staticmethod
    def forward(ctx, x, threshold):
        # Scale to [-1, 0, 1]
        # BitNet uses abs mean scaling then rounds to nearest {-1, 0, 1}
        ctx.save_for_backward(x, threshold)
        y = torch.sign(x)
        y[x.abs() < 0.5] = 0.0
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x.abs() > threshold] = 0.0
        return grad_x, None


class BitNetTernaryLinear(nn.Linear):
    """BitNet b1.58-style ternary quantization to {-1, 0, 1} with STE.

    Reference: "BitNet: Scaling 1-bit Transformers for Large Language Models"
    and "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits".
    """

    def __init__(self, in_features, out_features, bias=False, threshold=1.0, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
        self.threshold = threshold

    def _ternary_quantize(self, weight):
        gamma = weight.abs().mean().clamp(min=1e-8).detach()
        scaled = weight / gamma
        quantized = BitNetTernaryFunction.apply(scaled, torch.tensor(self.threshold, device=weight.device, dtype=weight.dtype))
        return quantized * gamma

    def forward(self, x):
        qweight = self._ternary_quantize(self.weight)
        return F.linear(x, qweight, self.bias)


class Uniform2BitFunction(torch.autograd.Function):
    """STE backward for uniform 2-bit quantization to {-a, -a/3, a/3, a}."""

    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, threshold)

        # 4 uniform levels: {-1, -1/3, 1/3, 1}
        # Boundaries at -2/3, 0, 2/3
        levels = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=x.device, dtype=x.dtype)

        # Find nearest level for each element
        flat = x.view(-1, 1)
        dist = (flat - levels.unsqueeze(0)).abs()
        indices = dist.argmin(dim=1)
        y = levels[indices].view(x.shape)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x.abs() > threshold] = 0.0
        return grad_x, None


class Uniform2BitLinear(nn.Linear):
    """Uniform 2-bit quantization to 4 evenly-spaced levels with STE.

    States: {-alpha, -alpha/3, alpha/3, alpha}
    This is the naive 2-bit baseline that does NOT use power-of-two shifting.
    """

    def __init__(self, in_features, out_features, bias=False, threshold=1.0, **kwargs):
        super().__init__(in_features, out_features, bias=bias)
        self.threshold = threshold

    def _uniform_quantize(self, weight):
        gamma = weight.abs().max().clamp(min=1e-8).detach()  # use max for uniform
        scaled = weight / gamma
        quantized = Uniform2BitFunction.apply(scaled, torch.tensor(self.threshold, device=weight.device, dtype=weight.dtype))
        return quantized * gamma

    def forward(self, x):
        qweight = self._uniform_quantize(self.weight)
        return F.linear(x, qweight, self.bias)
