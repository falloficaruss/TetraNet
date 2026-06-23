import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from quaternary import QBitLinearQuaternary, quaternary_weight_quantize

def test_quaternary_quantize_forward():
    print("--- Testing Forward Pass & Shape Handling ---")
    c = torch.tensor(0.5)
    
    # Create test weights ranging from -1.5 to 1.5
    # gamma = 1.0 (mean of abs is 1.0)
    weight = torch.tensor([[-1.5, -0.75, -0.25, 0.25, 0.75, 1.5]])
    
    # Boundary is (1 + 0.5) / 2 = 0.75
    # Expected quantization mapping (scaled by gamma=0.833):
    # -1.5/0.833 = -1.8 -> -1
    # -0.75/0.833 = -0.9 -> -1
    # -0.25/0.833 = -0.3 -> -0.5 (-c)
    # 0.25/0.833 = 0.3 -> 0.5 (c)
    # 0.75/0.833 = 0.9 -> 1
    # 1.5/0.833 = 1.8 -> 1
    
    quantized_weight = quaternary_weight_quantize(weight, c)
    print("Original Weights:", weight)
    print("Quantized Weights:", quantized_weight)
    
    assert weight.shape == quantized_weight.shape, "Shape mismatch after quantization"
    print("Shape handling: OK")
    print()

def test_gradient_flow():
    print("--- Testing Gradient Flow (STE & analytical `c` gradient) ---")
    torch.manual_seed(42)
    
    # Initialize bit linear layer
    layer = QBitLinearQuaternary(in_features=4, out_features=2, initial_c=0.375)
    
    # Ensure gradients are enabled
    layer.weight.requires_grad = True
    
    print("Initial c:", layer.c.item())
    
    # Forward pass with random inputs
    x = torch.randn(8, 4)
    y_pred = layer(x)
    
    # Target tensor
    y_target = torch.randn(8, 2)
    
    # Compute loss
    loss = nn.MSELoss()(y_pred, y_target)
    
    # Backward pass
    loss.backward()
    
    # Verify gradients
    print("Weight Gradient exists:", layer.weight.grad is not None)
    print("Weight Gradient norm:", layer.weight.grad.norm().item() if layer.weight.grad is not None else 0.0)
    print("c Gradient exists:", layer.c.grad is not None)
    print("c Gradient value:", layer.c.grad.item() if layer.c.grad is not None else 0.0)
    
    assert layer.weight.grad is not None, "STE failed: No gradient for weights"
    assert layer.c.grad is not None, "Gradient flow to c failed"
    print("Gradient flow: OK")
    print()

def run_all():
    test_quaternary_quantize_forward()
    test_gradient_flow()
    print("All tests passed successfully.")

if __name__ == "__main__":
    run_all()
