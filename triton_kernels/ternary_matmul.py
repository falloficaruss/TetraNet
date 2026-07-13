import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _ternary_kernel(
        a_ptr, w_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            for ki in range(BLOCK_K):
                k = k0 + ki
                if k >= K:
                    break
                w_k = tl.load(
                    w_ptr + k * stride_wk + offs_n * stride_wn,
                    mask=mask_n, other=0,
                )
                a_k = tl.load(
                    a_ptr + offs_m * stride_am + k * stride_ak,
                    mask=mask_m, other=0,
                )
                a_col = a_k[:, None]
                w_row = w_k[None, :]
                acc += tl.where(w_row == 1, a_col, 0)
                acc -= tl.where(w_row == -1, a_col, 0)

        c = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c, acc, mask=mask_m[:, None] & mask_n[None, :])


def ternary_matmul(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_w, N = w.shape
    assert K == K_w, f"dim mismatch: A({K}) != W({K_w})"
    c = torch.zeros(M, N, dtype=torch.int32, device=a.device)

    if not _HAS_TRITON or a.device.type != "cuda":
        raise RuntimeError("triton ternary_matmul requires CUDA + triton")

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _ternary_kernel[grid](
        a, w, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
