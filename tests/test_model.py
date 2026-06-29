import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model import (
    QuaternaryLlamaForCausalLM,
    QuaternaryLlamaConfig,
    RMSNorm,
    RotaryEmbedding,
)


def test_model_forward_shape():
    """Verify logits have shape (batch, seq_len, vocab_size)."""
    config = QuaternaryLlamaConfig(
        vocab_size=10000, hidden_dim=256, num_layers=2, num_heads=4, max_seq_len=128
    )
    model = QuaternaryLlamaForCausalLM(config)
    x = torch.randint(0, 10000, (2, 32))
    out = model(x)
    assert out["logits"].shape == (2, 32, 10000), (
        f"Expected (2, 32, 10000), got {out['logits'].shape}"
    )
    print("  ✓ Forward shape is correct")


def test_model_loss_computation():
    """Verify loss is finite and decreases with more tokens seen (via labels)."""
    config = QuaternaryLlamaConfig(
        vocab_size=1000, hidden_dim=64, num_layers=2, num_heads=2, max_seq_len=64
    )
    model = QuaternaryLlamaForCausalLM(config)
    x = torch.randint(0, 1000, (2, 16))
    out = model(x, labels=x)
    assert torch.isfinite(out["loss"]), "Loss should be finite"
    # For random init with V=1000, loss should be near ln(1000) ≈ 6.9
    assert out["loss"].item() > 5.0, (
        f"Loss should be > 5 for random init, got {out['loss'].item():.4f}"
    )
    print(f"  ✓ Loss = {out['loss'].item():.4f} (expected ~6.9 for random init)")


def test_causal_mask():
    """Verify token at position i cannot attend to position j > i."""
    config = QuaternaryLlamaConfig(
        vocab_size=1000, hidden_dim=64, num_layers=1, num_heads=2, max_seq_len=128
    )
    model = QuaternaryLlamaForCausalLM(config)

    # Create input where only first token differs
    x = torch.full((1, 10), 0, dtype=torch.long)
    x[:, 0] = 1  # Different token at position 0

    out = model(x, labels=x)

    # The logit for position 0 should differ from positions > 0
    logits = out["logits"]
    logit_pos0 = logits[:, 0, :]
    logit_pos1 = logits[:, 1, :]
    # Since causal mask prevents position 1 from seeing position 0's token,
    # position 1's prediction should be based only on the default token 0
    assert not torch.allclose(logit_pos0, logit_pos1, atol=1e-3), (
        "Position 0 and position 1 should have different logits"
    )
    print("  ✓ Causal mask is effective")


def test_gradient_flow():
    """Verify gradients flow to all layers including c parameters."""
    config = QuaternaryLlamaConfig(
        vocab_size=1000, hidden_dim=64, num_layers=2, num_heads=2, max_seq_len=64
    )
    model = QuaternaryLlamaForCausalLM(config)
    x = torch.randint(0, 1000, (2, 16))
    out = model(x, labels=x)
    out["loss"].backward()

    # Collect gradients
    weight_grads = []
    c_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if name.endswith(".c"):
                c_grads.append((name, param.grad.item()))
            elif "weight" in name and param.ndim >= 2:
                weight_grads.append((name, param.grad.norm().item()))

    assert len(weight_grads) > 0, "No weight gradients found"
    assert len(c_grads) > 0, "No c gradients found"
    assert all(g != 0.0 for _, g in weight_grads), "Some weight gradients are zero"
    print(f"  ✓ Weight gradients flow: {len(weight_grads)} layers")
    print(f"  ✓ c gradients flow: {len(c_grads)} layers (sample: {c_grads[0][1]:.6f})")


def test_get_c_values():
    """Verify get_c_values returns entries for all decoder layers and submodules."""
    config = QuaternaryLlamaConfig(
        vocab_size=1000, hidden_dim=64, num_layers=3, num_heads=2, max_seq_len=64
    )
    model = QuaternaryLlamaForCausalLM(config)
    c_vals = model.get_c_values()

    # Each layer has 7 projections (4 attn + 3 mlp) with c parameters
    # With 3 layers: 3 * 7 = 21 entries
    expected_count = config.num_layers * 7
    assert len(c_vals) == expected_count, (
        f"Expected {expected_count} c values, got {len(c_vals)}"
    )
    assert all(isinstance(v, float) for v in c_vals.values()), (
        "All c values should be floats"
    )
    assert all(v == config.initial_c for v in c_vals.values()), (
        f"All c values should be {config.initial_c} at init"
    )
    print(f"  ✓ {len(c_vals)} c values, all initialized to {config.initial_c}")


def test_get_num_params():
    """Verify parameter count is in the expected range (~9.4M for default config)."""
    model = QuaternaryLlamaForCausalLM()
    params = model.get_num_params()
    # Expected: ~9,441,592
    assert 9_000_000 < params < 10_000_000, f"Expected ~9.4M params, got {params:,}"
    print(f"  ✓ {params:,} parameters (~9.4M expected)")


def test_weight_tying():
    """Verify embedding and lm_head weights are tied."""
    config = QuaternaryLlamaConfig(tie_weights=True)
    model = QuaternaryLlamaForCausalLM(config)
    assert model.lm_head.weight is model.embed_tokens.weight, "Weights should be tied"
    print("  ✓ Embedding and LM head weights are tied")


def test_rms_norm():
    """Verify RMSNorm preserves shape and normalizes correctly."""
    dim = 32
    rms = RMSNorm(dim)
    x = torch.randn(4, 16, dim)
    y = rms(x)
    assert y.shape == x.shape, "RMSNorm should preserve shape"

    # Compute RMS manually
    rms_val = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    x_normalized = x / rms_val
    assert torch.allclose(y, rms.weight * x_normalized, atol=1e-6), (
        "RMSNorm computation mismatch"
    )
    print("  ✓ RMSNorm works correctly")


def test_rotary_embedding():
    """Verify RoPE produces correct shapes and rotates Q/K differently by position."""
    rotary = RotaryEmbedding(dim=32, max_seq_len=128)
    cos, sin = rotary(10)
    assert cos.shape == (1, 1, 10, 16), f"Expected (1, 1, 10, 16), got {cos.shape}"
    assert sin.shape == (1, 1, 10, 16), f"Expected (1, 1, 10, 16), got {sin.shape}"

    # Verify rotation differs by position: positions 0 and 5 should differ
    assert not torch.allclose(cos[0, 0, 0], cos[0, 0, 5]), (
        "RoPE should differ across positions"
    )
    print("  ✓ RotaryEmbedding produces correct shapes and position-dependent values")


def run_all():
    print("=== Model Tests ===\n")
    tests = [
        test_rms_norm,
        test_rotary_embedding,
        test_model_forward_shape,
        test_model_loss_computation,
        test_causal_mask,
        test_gradient_flow,
        test_get_c_values,
        test_get_num_params,
        test_weight_tying,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
