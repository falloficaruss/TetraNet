import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from quaternary import FixedCQuaternaryLinear, quaternary_weight_quantize


def test_quaternary_quantize_forward():
    print("--- Testing Forward Pass & Shape Handling ---")
    c = torch.tensor(0.5)

    weight = torch.tensor([[-1.5, -0.75, -0.25, 0.25, 0.75, 1.5]])

    quantized_weight = quaternary_weight_quantize(weight, c)
    print("Original Weights:", weight)
    print("Quantized Weights:", quantized_weight)

    assert weight.shape == quantized_weight.shape, "Shape mismatch after quantization"
    print("Shape handling: OK")
    print()


def test_ste_gradient_flow():
    """STE gradient should flow to weights (no c grad since c is a buffer)."""
    print("--- Testing STE Gradient Flow ---")
    torch.manual_seed(42)

    layer = FixedCQuaternaryLinear(in_features=4, out_features=2, initial_c=0.375)
    layer.weight.requires_grad = True

    x = torch.randn(8, 4)
    y_pred = layer(x)
    y_target = torch.randn(8, 2)

    loss = torch.nn.MSELoss()(y_pred, y_target)
    loss.backward()

    assert layer.weight.grad is not None, "STE failed: No gradient for weights"
    print("Weight gradient norm:", layer.weight.grad.norm().item())
    print("c is buffer (no learnable gradient):", not layer.c.requires_grad)
    print("STE gradient flow: OK")
    print()


def run_all():
    test_quaternary_quantize_forward()
    test_ste_gradient_flow()
    print("All tests passed successfully.")


if __name__ == "__main__":
    run_all()
