"""Correctness tests for specialized inference kernels vs dense reference."""

import torch
import torch.nn.functional as F

from specialized import (
    bitnet_quantize_weight,
    uniform_quantize_weight,
    quaternary_quantize_weight,
    pack_ternary_int8,
    pack_quaternary_2bit,
    pack_uniform_2bit,
    quantize_activation_int8,
    ternary_matmul,
    shift_matmul,
    uniform_matmul,
    specialize_linear,
    specialize_model,
    backend_info,
    shift_bits_for_c,
)
from models.layers import BitNetTernaryLinear, Uniform2BitLinear, FullPrecisionLinear
from quaternary import FixedCQuaternaryLinear


def test_backend_import():
    info = backend_info()
    print("backends:", info)
    assert "cpu_ext" in info


def test_ternary_matches_dense():
    torch.manual_seed(0)
    M, K, N = 4, 32, 16
    W = torch.randn(N, K)  # [out, in]
    q, gamma = bitnet_quantize_weight(W)
    x = torch.randn(M, K)
    a, act_scale = quantize_activation_int8(x)

    codes = pack_ternary_int8(q.T.contiguous())
    y_k = ternary_matmul(a, codes).float() * act_scale * gamma

    # dense with same quantized acts and weights
    x_q = a.float() * act_scale
    y_ref = x_q @ (q * gamma).T

    err = (y_k - y_ref).abs().max().item()
    print(f"ternary max err: {err:.6e}")
    assert err < 1e-3, err


def test_shift_matches_dense():
    torch.manual_seed(1)
    M, K, N = 4, 32, 16
    c = 0.5
    W = torch.randn(N, K)
    q, gamma = quaternary_quantize_weight(W, c)
    x = torch.randn(M, K)
    a, act_scale = quantize_activation_int8(x)

    packed = pack_quaternary_2bit(q.T.contiguous().float(), c)
    sb = shift_bits_for_c(c)
    y_k = shift_matmul(a, packed, sb, K).float() * act_scale * gamma

    x_q = a.float() * act_scale
    y_ref = x_q @ (q * gamma).T

    # shift on int acts is exact for c in {0.25,0.5} only when act is int;
    # compare against integer shift reference
    w_t = q.T  # [K,N] in {-1,-c,c,1}
    y_shift_ref = torch.zeros(M, N)
    for k in range(K):
        ak = a[:, k].float()
        for n in range(N):
            v = w_t[k, n].item()
            if abs(v - 1.0) < 1e-5:
                term = ak
            elif abs(v + 1.0) < 1e-5:
                term = -ak
            elif abs(v - c) < 1e-5:
                term = (a[:, k] >> sb).float()
            else:
                term = -(a[:, k] >> sb).float()
            y_shift_ref[:, n] += term
    y_shift_ref = y_shift_ref * act_scale * gamma

    err = (y_k - y_shift_ref).abs().max().item()
    print(f"shift max err vs int-shift ref: {err:.6e}")
    assert err < 1e-4, err

    # loose check vs FP dense (truncation from integer shift)
    err_fp = (y_k - y_ref).abs().max().item() / (y_ref.abs().mean().item() + 1e-8)
    print(f"shift rel err vs FP dense: {err_fp:.4f}")
    assert err_fp < 0.15, err_fp


def test_uniform_matches_dense():
    torch.manual_seed(2)
    M, K, N = 4, 32, 16
    W = torch.randn(N, K)
    q, gamma = uniform_quantize_weight(W)
    x = torch.randn(M, K)
    a, act_scale = quantize_activation_int8(x)

    packed = pack_uniform_2bit(q.T.contiguous().float())
    y_k = uniform_matmul(a, packed, K) * act_scale * gamma

    x_q = a.float() * act_scale
    y_ref = x_q @ (q * gamma).T

    err = (y_k - y_ref).abs().max().item()
    print(f"uniform max err: {err:.6e}")
    assert err < 1e-3, err


def test_specialized_linear_shapes():
    torch.manual_seed(3)
    layer = FixedCQuaternaryLinear(64, 32, bias=False, initial_c=0.5)
    with torch.no_grad():
        layer.weight.normal_()
    spec = specialize_linear(layer, force="quaternary")
    x = torch.randn(2, 8, 64)
    y = spec(x)
    assert y.shape == (2, 8, 32), y.shape
    print("specialized shift shape OK", y.shape)


def test_specialize_tiny_model():
    from train_kaggle import build_model

    torch.manual_seed(4)
    model = build_model(
        baseline="fixed_c_05",
        vocab_size=128,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        max_seq_len=64,
    )
    specialize_model(model, baseline="fixed_c_05")
    ids = torch.randint(0, 128, (1, 8))
    out = model(ids)
    assert out["logits"].shape == (1, 8, 128)
    assert torch.isfinite(out["logits"]).all()
    print("tiny specialized model forward OK")


if __name__ == "__main__":
    test_backend_import()
    test_ternary_matches_dense()
    test_shift_matches_dense()
    test_uniform_matches_dense()
    test_specialized_linear_shapes()
    test_specialize_tiny_model()
    print("ALL PASSED")
