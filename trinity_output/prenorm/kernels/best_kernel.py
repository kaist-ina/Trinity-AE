import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32, 'BLOCK_L': 32}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_L': 64}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_L': 128}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_L': 32}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_L': 64}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_L': 128}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_L': 32}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_L': 64}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_L': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    const_1_ptr,
    const_1_stride0: tl.constexpr,
    const_1_stride1: tl.constexpr,
    const_2_ptr,
    const_2_stride0: tl.constexpr,
    const_2_stride1: tl.constexpr,
    const_3_ptr,
    const_3_stride0: tl.constexpr,
    const_3_stride1: tl.constexpr,
    const_4_ptr,
    const_4_stride0: tl.constexpr,
    const_4_stride1: tl.constexpr,
    const_4_stride2: tl.constexpr,
    const_5_ptr,
    const_5_stride0: tl.constexpr,
    const_5_stride1: tl.constexpr,
    const_5_stride2: tl.constexpr,
    lv37_ptr,
    lv37_stride0: tl.constexpr,
    lv37_stride1: tl.constexpr,
    x_ptr,
    x_stride0: tl.constexpr,
    x_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_L: tl.constexpr
):
    # Allocate intermediate tensors
    lv1 = tl.zeros((16,), dtype=tl.float16)
    lv30 = tl.zeros((1, 16, BLOCK_L), dtype=tl.float32)
    lv31 = tl.zeros((1, 16), dtype=tl.float32)
    lv35 = tl.zeros((1, 16, 128), dtype=tl.float32)
    lv5 = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    lv6 = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    lv7 = tl.zeros((16, BLOCK_N), dtype=tl.float32)

    # Parallel loop n from 0 to lv5_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * x_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * x_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < 4096)[None, :]
        temp_0 = tl.load(x_ptr + offset_0, mask=mask_0, other=0.0)
        lv1 = (tl.sum((temp_0 * temp_0).to(tl.float16), axis=1, dtype=tl.float16) + (1 * lv1).to(tl.float16)).to(tl.float16)
    # Skipped empty sloop with dummy body
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_1 = (tl.arange(0, 16))[:, None] * x_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * x_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_1 = (k_indices < 4096)[None, :]
        temp_1 = tl.load(x_ptr + offset_1, mask=mask_1, other=0.0)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * const_1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * const_1_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_2 = (k_indices < 4096)[:, None] & (n_indices < 4096)[None, :]
        temp_2 = tl.load(const_1_ptr + offset_2, mask=mask_2, other=0.0)
        lv5 = ((1 * lv5) + tl.dot(temp_1.to(tl.float32), temp_2.to(tl.float32)))
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * const_2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * const_2_stride1
        mask_3 = (k_indices < 4096)[:, None] & (n_indices < 4096)[None, :]
        temp_3 = tl.load(const_2_ptr + offset_3, mask=mask_3, other=0.0)
        lv6 = ((1 * lv6) + tl.dot(temp_1.to(tl.float32), temp_3.to(tl.float32)))
        offset_4 = (k + tl.arange(0, BLOCK_K))[:, None] * const_3_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * const_3_stride1
        mask_4 = (k_indices < 4096)[:, None] & (n_indices < 4096)[None, :]
        temp_4 = tl.load(const_3_ptr + offset_4, mask=mask_4, other=0.0)
        lv7 = ((1 * lv7) + tl.dot(temp_1.to(tl.float32), temp_4.to(tl.float32)))
    lv5 = (lv5 / tl.sqrt((lv1 / 4096).to(tl.float32))[:, None])
    lv6 = (lv6 / tl.sqrt((lv1 / 4096).to(tl.float32))[:, None])
    lv7 = (lv7 / tl.sqrt((lv1 / 4096).to(tl.float32))[:, None])
    temp_5 = tl.expand_dims(lv5, 1)
    lv11 = tl.permute(temp_5, (1, 0, 2))
    temp_6 = tl.expand_dims(lv6, 1)
    lv12 = tl.permute(temp_6, (1, 0, 2))
    temp_7 = tl.expand_dims(lv7, 1)
    lv13 = tl.permute(temp_7, (1, 0, 2))
    offset_5 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * const_4_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * const_4_stride1 + (tl.arange(0, 128))[None, None, :] * const_4_stride2
    elem_n_indices = ((n // BLOCK_N) + tl.arange(0, 1))
    mask_5 = (elem_n_indices < 32)[:, None, None]
    tl.store(const_4_ptr + offset_5, lv12.to(tl.float16), mask=mask_5)
    offset_6 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * const_5_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * const_5_stride1 + (tl.arange(0, 128))[None, None, :] * const_5_stride2
    mask_6 = (elem_n_indices < 32)[:, None, None]
    tl.store(const_5_ptr + offset_6, lv13.to(tl.float16), mask=mask_6)
    # Sequential loop l from 0 to 1024 with tile size BLOCK_L
    for l in range(0, 1024, BLOCK_L):
        offset_7 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * const_4_stride0 + (l + tl.arange(0, BLOCK_L))[None, :, None] * const_4_stride1 + (tl.arange(0, 128))[None, None, :] * const_4_stride2
        elem_n_indices = ((n // BLOCK_N) + tl.arange(0, 1))
        l_indices = l + tl.arange(0, BLOCK_L)
        mask_7 = (elem_n_indices < 32)[:, None, None] & (l_indices < 1024)[None, :, None]
        temp_8 = tl.load(const_4_ptr + offset_7, mask=mask_7, other=0.0)
        temp_9 = tl.permute(temp_8, (0, 2, 1))
        lv30 = tl.exp(tl.dot(lv11, temp_9.to(tl.float32)).to(tl.float32))
        lv31 = (tl.sum(lv30, axis=2, dtype=tl.float32) + (1 * lv31))
        offset_8 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * const_5_stride0 + (l + tl.arange(0, BLOCK_L))[None, :, None] * const_5_stride1 + (tl.arange(0, 128))[None, None, :] * const_5_stride2
        mask_8 = (elem_n_indices < 32)[:, None, None] & (l_indices < 1024)[None, :, None]
        temp_10 = tl.load(const_5_ptr + offset_8, mask=mask_8, other=0.0)
        lv35 = ((1 * lv35) + tl.dot(lv30, temp_10.to(tl.float32)))
    # Skipped empty sloop with dummy body
    lv35 = (lv35 / lv31[:, :, None])
    temp_11 = tl.permute(lv35, (1, 0, 2))
    offset_9 = (tl.arange(0, 16))[:, None] * lv37_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * lv37_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_9 = (n_indices < 4096)[None, :]
    tl.store(lv37_ptr + offset_9, tl.reshape(temp_11, (16, 128)).to(tl.float16), mask=mask_9)


# Metadata for benchmark.py
TENSOR_PARAMS = ['const_1', 'const_2', 'const_3', 'const_4', 'const_5', 'lv37', 'x']
FP32_TENSOR_PARAMS = []
BLOCK_PARAMS = ['block_k', 'block_l']

def forward(const_1, const_2, const_3, const_4, const_5, lv37, x, block_k=16, block_l=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4096 - 0 + 128 - 1) // 128,)](
        const_1,
        const_1.stride(0),
        const_1.stride(1),
        const_2,
        const_2.stride(0),
        const_2.stride(1),
        const_3,
        const_3.stride(0),
        const_3.stride(1),
        const_4,
        const_4.stride(0),
        const_4.stride(1),
        const_4.stride(2),
        const_5,
        const_5.stride(0),
        const_5.stride(1),
        const_5.stride(2),
        lv37,
        lv37.stride(0),
        lv37.stride(1),
        x,
        x.stride(0),
        x.stride(1),
        # BLOCK_K, BLOCK_L are provided by autotune,
        BLOCK_N=128,
        # BLOCK_K is automatically set by autotune,
        # BLOCK_L is automatically set by autotune
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
