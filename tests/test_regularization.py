import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from quaternary import QBitLinearQuaternary
from regularization import (
    multi_well_potential,
    AdaptiveSnappingScheduler,
    compute_total_loss,
)
from model import QuaternaryLlamaForCausalLM, QuaternaryLlamaConfig


def test_penalty_zero_at_wells():
    """c exactly at well positions → penalty ≈ 0."""
    model = nn.ModuleList(
        [
            QBitLinearQuaternary(4, 4, initial_c=0.25),
            QBitLinearQuaternary(4, 4, initial_c=0.5),
        ]
    )
    penalty = multi_well_potential(model)
    assert penalty.item() == 0.0, f"Expected 0.0, got {penalty.item()}"


def test_penalty_positive_between_wells():
    """c between wells (midpoint 0.375) → penalty > 0."""
    model = nn.ModuleList([QBitLinearQuaternary(4, 4, initial_c=0.375)])
    penalty = multi_well_potential(model)
    # |0.375 - 0.25| = 0.125, |0.375 - 0.5| = 0.125, min = 0.125
    assert penalty.item() == 0.125, f"Expected 0.125, got {penalty.item()}"


def test_penalty_gradient_flow():
    """Backward through penalty produces non-zero gradients on c."""
    model = nn.ModuleList([QBitLinearQuaternary(4, 4, initial_c=0.375)])
    penalty = multi_well_potential(model)
    penalty.backward()

    assert model[0].c.grad is not None, "c gradient should not be None"
    assert model[0].c.grad.item() != 0.0, "c gradient should be non-zero"


def test_scheduler_returns_zero_before_snap():
    """Progress ≤ 0 → lambda = 0."""
    scheduler = AdaptiveSnappingScheduler(alpha=0.02, snap_start=0.9)
    task_loss = torch.tensor(5.0)

    lam_negative = scheduler.get_lambda(-0.1, task_loss)
    lam_zero = scheduler.get_lambda(0.0, task_loss)

    assert lam_negative == 0.0, f"Expected 0.0, got {lam_negative}"
    assert lam_zero == 0.0, f"Expected 0.0, got {lam_zero}"


def test_scheduler_self_adaptive():
    """Lambda equals alpha * task_loss * progress."""
    scheduler = AdaptiveSnappingScheduler(alpha=0.02, snap_start=0.9)
    task_loss = torch.tensor(4.0)

    lam = scheduler.get_lambda(0.5, task_loss)

    expected = 0.02 * 4.0 * 0.5
    assert lam == expected, f"Expected {expected}, got {lam}"


def test_scheduler_respects_alpha():
    """Doubling alpha doubles lambda."""
    task_loss = torch.tensor(3.0)
    progress = 0.75

    s1 = AdaptiveSnappingScheduler(alpha=0.01)
    s2 = AdaptiveSnappingScheduler(alpha=0.02)

    lam1 = s1.get_lambda(progress, task_loss)
    lam2 = s2.get_lambda(progress, task_loss)

    assert lam2 == 2.0 * lam1, f"Expected {2 * lam1}, got {lam2}"


def test_total_loss_end_to_end():
    """Full model: total loss = task_loss + lambda * penalty, gradients flow to c."""
    config = QuaternaryLlamaConfig(
        vocab_size=1000,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        max_seq_len=64,
    )
    model = QuaternaryLlamaForCausalLM(config)
    scheduler = AdaptiveSnappingScheduler(alpha=0.02, snap_start=0.9)

    x = torch.randint(0, 1000, (2, 16))
    out = model(x, labels=x)
    task_loss = out["loss"]

    total_loss = compute_total_loss(
        task_loss, model, progress=0.95, scheduler=scheduler
    )

    total_loss.backward()

    # Verify c gradients are non-zero
    c_grads = []
    for _name, param in model.named_parameters():
        if param.grad is not None and _name.endswith(".c"):
            c_grads.append(param.grad.item())

    assert len(c_grads) > 0, "No c gradients found after backward"
    assert all(g != 0.0 for g in c_grads), "Some c gradients are zero"

    # Total loss should be strictly greater than task loss (positive penalty + lambda)
    assert total_loss.item() > task_loss.item(), (
        f"Total loss {total_loss.item():.4f} should exceed task loss {task_loss.item():.4f}"
    )


def run_all():
    print("=== Regularization Tests ===\n")
    tests = [
        test_penalty_zero_at_wells,
        test_penalty_positive_between_wells,
        test_penalty_gradient_flow,
        test_scheduler_returns_zero_before_snap,
        test_scheduler_self_adaptive,
        test_scheduler_respects_alpha,
        test_total_loss_end_to_end,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_all()
