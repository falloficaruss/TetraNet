import torch
import torch.nn as nn
import torch.nn.functional as F


class QuaternaryQuantizeFunction(torch.autograd.Function):
    """
    Core kernel for mapping scaled weights to the quaternary grid {-1, -c, c, 1}.
    Implements a custom backward pass for Straight-Through Estimator (STE) on weights
    and analytical gradients for the intermediate state parameter 'c'.
    """

    @staticmethod
    def forward(ctx, x, c, threshold):
        # States: {-1, -c, c, 1}
        # Midpoint between c and 1 is (1 + c) / 2
        # Midpoint between -c and c is 0
        # Midpoint between -1 and -c is (-1 - c) / 2 = -(1 + c) / 2
        boundary = (1.0 + c) / 2.0

        y = torch.ones_like(x)
        y = torch.where(x <= boundary, c * torch.ones_like(x), y)
        y = torch.where(x <= 0.0, -c * torch.ones_like(x), y)
        y = torch.where(x <= -boundary, -torch.ones_like(x), y)

        ctx.save_for_backward(x, c, threshold)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, c, threshold = ctx.saved_tensors
        boundary = (1.0 + c) / 2.0

        # 1. Gradient for x (Straight-Through Estimator)
        # Pass gradients straight through, but clip if magnitude exceeds threshold
        grad_x = grad_output.clone()
        grad_x[x.abs() > threshold] = 0.0

        # 2. Gradient for c
        # The derivative of the quantization function w.r.t c:
        # y = c if x in (0, boundary] -> dy/dc = 1
        # y = -c if x in (-boundary, 0] -> dy/dc = -1
        # otherwise -> dy/dc = 0
        grad_c_mask = torch.zeros_like(x)
        grad_c_mask = torch.where(
            (x > 0.0) & (x <= boundary), torch.ones_like(x), grad_c_mask
        )
        grad_c_mask = torch.where(
            (x <= 0.0) & (x > -boundary), -torch.ones_like(x), grad_c_mask
        )

        # Sum over all weight elements to get the scalar gradient for c
        grad_c = (grad_output * grad_c_mask).sum()

        # Return gradients for x, c, and None for threshold
        return grad_x, grad_c, None


def quaternary_weight_quantize(weight, c, threshold=1.0):
    """
    Applies layer-wise scaling, quantization to {-1, -c, c, 1}, and re-scaling.
    """
    # 1. Calculate scaling factor gamma (mean of absolute values)
    # Detach to prevent gradients flowing through the scaling factor (standard practice in 1-bit/2-bit LLMs)
    gamma = weight.abs().mean().clamp(min=1e-8).detach()

    # 2. Scale weights
    scaled_weight = weight / gamma

    # 3. Apply Quaternary Quantization with custom STE
    quantized_scaled_weight = QuaternaryQuantizeFunction.apply(
        scaled_weight,
        c,
        torch.tensor(threshold, dtype=weight.dtype, device=weight.device),
    )

    # 4. Scale back
    quantized_weight = quantized_scaled_weight * gamma

    return quantized_weight


class QBitLinearQuaternary(nn.Linear):
    """
    A custom Linear layer that quantizes weights to the quaternary grid during the forward pass.
    """

    def __init__(
        self, in_features, out_features, bias=False, initial_c=0.375, threshold=1.0
    ):
        super().__init__(in_features, out_features, bias=bias)
        # Learnable floating-point parameter c for this layer
        self.c = nn.Parameter(torch.tensor(initial_c))
        self.threshold = threshold

    def forward(self, x):
        # Quantize weights
        quantized_weight = quaternary_weight_quantize(
            self.weight, self.c, self.threshold
        )

        # Standard linear projection using quantized weights
        return F.linear(x, quantized_weight, self.bias)
