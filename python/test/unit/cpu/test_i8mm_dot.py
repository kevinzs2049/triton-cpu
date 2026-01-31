import pytest
import torch
import triton
import triton.language as tl


def _is_cpu():
    return triton.runtime.driver.active.get_current_target().backend == "cpu"


@triton.jit
def i8_dot_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        acc += tl.dot(a, b)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@pytest.mark.cpu
@pytest.mark.parametrize(
    "M,N,K",
    [
        (4, 4, 8),
        (4, 4, 16),
        (8, 8, 16),
        (16, 16, 16),
        (16, 16, 32),
        (32, 16, 16),
        (16, 32, 16),
    ],
)
def test_cpu_i8_dot_smmla(M, N, K):
    if not _is_cpu():
        pytest.skip("CPU backend only")
    triton.runtime.driver.set_active_to_cpu()
    torch.manual_seed(0)

    a = torch.randint(-3, 3, (M, K), dtype=torch.int8, device="cpu")
    b = torch.randint(-3, 3, (K, N), dtype=torch.int8, device="cpu")
    c = torch.empty((M, N), dtype=torch.int32, device="cpu")

    grid = (triton.cdiv(M, 4), triton.cdiv(N, 4))
    i8_dot_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=4, BLOCK_N=4, BLOCK_K=8,
    )

    ref = a.to(torch.int32) @ b.to(torch.int32)
    assert torch.equal(c, ref)
