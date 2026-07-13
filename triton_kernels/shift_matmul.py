import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _shift_kernel(
        a_ptr, w_packed_ptr, c_ptr,
        M, N, K, K_packed, shift_bits,
        stride_am, stride_ak,
        stride_wp_pk, stride_wp_n,
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

        for pk0 in range(0, K_packed, BLOCK_K):
            for pk_i in range(BLOCK_K):
                pk = pk0 + pk_i
                if pk >= K_packed:
                    break
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
                    )
                    a_col = a_k[:, None]
                    code_row = code[None, :]
                    acc += tl.where(code_row == 0, a_col, 0)
                    acc -= tl.where(code_row == 1, a_col, 0)
                    acc += tl.where(code_row == 2, a_col >> shift_bits, 0)
                    acc -= tl.where(code_row == 3, a_col >> shift_bits, 0)

        c = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c, acc, mask=mask_m[:, None] & mask_n[None, :])


def shift_matmul(a: torch.Tensor, w_packed: torch.Tensor, shift_bits: int) -> torch.Tensor:
    M, K = a.shape
    K_packed, N = w_packed.shape
    c = torch.zeros(M, N, dtype=torch.int32, device=a.device)

    if not _HAS_TRITON or a.device.type != "cuda":
        # CPU / no-triton path should use specialized.py fallback
        raise RuntimeError("triton shift_matmul requires CUDA + triton")

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 8
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _shift_kernel[grid](
        a, w_packed, c,
        M, N, K, K_packed, shift_bits,
        a.stride(0), a.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c
