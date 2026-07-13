import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _uniform_kernel(
        a_ptr, w_packed_ptr, c_ptr,
        M, N, K, K_packed,
        stride_am, stride_ak,
        stride_wp_pk, stride_wp_n,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for pk in range(K_packed):
            k_base = pk * 4
            wp = tl.load(
                w_packed_ptr + pk * stride_wp_pk + offs_n * stride_wp_n,
                mask=mask_n, other=0,
            ).to(tl.int32)

            for b in range(4):
                k = k_base + b
                if k >= K:
                    break
                code = (wp >> (b * 2)) & 3
                a_k = tl.load(
                    a_ptr + offs_m * stride_am + k * stride_ak,
                    mask=mask_m, other=0,
                ).to(tl.float32)
                a_col = a_k[:, None]
                code_row = code[None, :]
                # levels: 0=-1, 1=-1/3, 2=+1/3, 3=+1
                lvl = tl.where(code_row == 0, -1.0,
                      tl.where(code_row == 1, -1.0 / 3.0,
                      tl.where(code_row == 2, 1.0 / 3.0, 1.0)))
                acc += a_col * lvl

        c = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c, acc, mask=mask_m[:, None] & mask_n[None, :])


def uniform_matmul(a: torch.Tensor, w_packed: torch.Tensor, K: int) -> torch.Tensor:
    M, K_a = a.shape
    assert K_a == K
    K_packed, N = w_packed.shape
    c = torch.zeros(M, N, dtype=torch.float32, device=a.device)

    if not _HAS_TRITON or a.device.type != "cuda":
        raise RuntimeError("triton uniform_matmul requires CUDA + triton")

    BLOCK_M, BLOCK_N = 32, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _uniform_kernel[grid](
        a, w_packed, c,
        M, N, K, K_packed,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return c
