import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_M': 128}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_M': 128}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_M': 32}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_M': 64}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_M': 128})
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
    const_6_ptr,
    const_6_stride0: tl.constexpr,
    const_6_stride1: tl.constexpr,
    const_6_stride2: tl.constexpr,
    lv24_ptr,
    lv24_stride0: tl.constexpr,
    lv24_stride1: tl.constexpr,
    lv24_stride2: tl.constexpr,
    lv27_ptr,
    lv27_stride0: tl.constexpr,
    lv27_stride1: tl.constexpr,
    lv27_stride2: tl.constexpr,
    lv28_ptr,
    lv28_stride0: tl.constexpr,
    lv28_stride1: tl.constexpr,
    lv28_stride2: tl.constexpr,
    lv29_ptr,
    lv29_stride0: tl.constexpr,
    lv29_stride1: tl.constexpr,
    lv30_ptr,
    lv30_stride0: tl.constexpr,
    lv30_stride1: tl.constexpr,
    x_ptr,
    x_stride0: tl.constexpr,
    x_stride1: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    # Allocate intermediate tensors
    lv = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    lv1 = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    lv2 = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    lv6 = tl.zeros((1, 16, 128), dtype=tl.float16)
    lv7 = tl.zeros((1, 16, 128), dtype=tl.float16)
    lv8 = tl.zeros((1, 16, 128), dtype=tl.float16)

    # Initialize kernel accumulators
    lv29 = tl.zeros((1, 16), dtype=tl.float32)
    lv30 = tl.zeros((1, 16), dtype=tl.float32)
    # Parallel loop j from 0 to lv_dim1 with tile size BLOCK_J
    # Executed across grid dimension 0
    j = 0 + tl.program_id(0) * BLOCK_J
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * x_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * x_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < 4096)[None, :]
        temp_0 = tl.load(x_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * const_1_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * const_1_stride1
        j_indices = j + tl.arange(0, BLOCK_J)
        mask_1 = (k_indices < 4096)[:, None] & (j_indices < 4096)[None, :]
        temp_1 = tl.load(const_1_ptr + offset_1, mask=mask_1, other=0.0)
        lv = ((1 * lv).to(tl.float16) + tl.dot(temp_0, temp_1).to(tl.float16)).to(tl.float16)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * const_2_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * const_2_stride1
        mask_2 = (k_indices < 4096)[:, None] & (j_indices < 4096)[None, :]
        temp_2 = tl.load(const_2_ptr + offset_2, mask=mask_2, other=0.0)
        lv1 = ((1 * lv1).to(tl.float16) + tl.dot(temp_0, temp_2).to(tl.float16)).to(tl.float16)
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * const_3_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * const_3_stride1
        mask_3 = (k_indices < 4096)[:, None] & (j_indices < 4096)[None, :]
        temp_3 = tl.load(const_3_ptr + offset_3, mask=mask_3, other=0.0)
        lv2 = ((1 * lv2).to(tl.float16) + tl.dot(temp_0, temp_3).to(tl.float16)).to(tl.float16)

    lv3 = tl.expand_dims(lv, 1)

    lv4 = tl.expand_dims(lv1, 1)

    lv5 = tl.expand_dims(lv2, 1)
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        lv6 = tl.permute(lv3, (1, 0, 2))
        lv7 = tl.permute(lv4, (1, 0, 2))
        lv8 = tl.permute(lv5, (1, 0, 2))
        offset_4 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_4_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * const_4_stride1 + (tl.arange(0, 128))[None, None, :] * const_4_stride2
        elem_j_indices = ((j // BLOCK_J) + tl.arange(0, 1))
        mask_4 = (elem_j_indices < 32)[:, None, None]
        tl.store(const_4_ptr + offset_4, lv7.to(tl.float16), mask=mask_4)
        offset_5 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_5_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * const_5_stride1 + (tl.arange(0, 128))[None, None, :] * const_5_stride2
        mask_5 = (elem_j_indices < 32)[:, None, None]
        tl.store(const_5_ptr + offset_5, lv8.to(tl.float16), mask=mask_5)
        offset_6 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_4_stride0 + (m + tl.arange(0, BLOCK_M))[None, :, None] * const_4_stride1 + (tl.arange(0, 128))[None, None, :] * const_4_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_6 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, :, None]
        temp_4 = tl.load(const_4_ptr + offset_6, mask=mask_6, other=0.0)
        temp_5 = tl.permute(temp_4, (0, 2, 1))
        offset_7 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv24_stride0 + (tl.arange(0, 16))[None, :, None] * lv24_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv24_stride2
        mask_7 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        tl.store(lv24_ptr + offset_7, tl.dot(lv6.to(tl.float32), temp_5.to(tl.float32)).to(tl.float32), mask=mask_7)
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        offset_8 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv24_stride0 + (tl.arange(0, 16))[None, :, None] * lv24_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv24_stride2
        elem_j_indices = ((j // BLOCK_J) + tl.arange(0, 1))
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_8 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        temp_6 = tl.load(lv24_ptr + offset_8, mask=mask_8, other=0.0)
        offset_9 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv27_stride0 + (tl.arange(0, 16))[None, :, None] * lv27_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv27_stride2
        mask_9 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        tl.store(lv27_ptr + offset_9, tl.exp(temp_6.to(tl.float32)).to(tl.float32), mask=mask_9)
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        offset_10 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_6_stride0 + (tl.arange(0, 16))[None, :, None] * const_6_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * const_6_stride2
        elem_j_indices = ((j // BLOCK_J) + tl.arange(0, 1))
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_10 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        temp_7 = tl.load(const_6_ptr + offset_10, mask=mask_10, other=0.0)
        offset_11 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv24_stride0 + (tl.arange(0, 16))[None, :, None] * lv24_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv24_stride2
        mask_11 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        temp_8 = tl.load(lv24_ptr + offset_11, mask=mask_11, other=0.0)
        offset_12 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv28_stride0 + (tl.arange(0, 16))[None, :, None] * lv28_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv28_stride2
        mask_12 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        tl.store(lv28_ptr + offset_12, tl.exp(((temp_7 + temp_8) / 1.5).to(tl.float32)).to(tl.float32), mask=mask_12)
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        offset_13 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv27_stride0 + (tl.arange(0, 16))[None, :, None] * lv27_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv27_stride2
        elem_j_indices = ((j // BLOCK_J) + tl.arange(0, 1))
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_13 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        temp_9 = tl.load(lv27_ptr + offset_13, mask=mask_13, other=0.0)
        lv29 = ((1 * lv29) + tl.sum(temp_9, axis=2, dtype=tl.float32))
        offset_14 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv28_stride0 + (tl.arange(0, 16))[None, :, None] * lv28_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv28_stride2
        mask_14 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        temp_10 = tl.load(lv28_ptr + offset_14, mask=mask_14, other=0.0)
        lv30 = ((1 * lv30) + tl.sum(temp_10, axis=2, dtype=tl.float32))
    # Store kernel accumulators
    offset_15 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None] * lv29_stride0 + (tl.arange(0, 16))[None, :] * lv29_stride1
    elem_j_indices = ((j // BLOCK_J) + tl.arange(0, 1))
    mask_15 = (elem_j_indices < 32)[:, None]
    tl.store(lv29_ptr + offset_15, lv29.to(tl.float16), mask=mask_15)
    offset_16 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None] * lv30_stride0 + (tl.arange(0, 16))[None, :] * lv30_stride1
    mask_16 = (elem_j_indices < 32)[:, None]
    tl.store(lv30_ptr + offset_16, lv30.to(tl.float16), mask=mask_16)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_M': 32}),
        triton.Config({'BLOCK_M': 64}),
        triton.Config({'BLOCK_M': 128})
    ], key=[]
)
@triton.jit
def kernel_1(
    const_5_ptr,
    const_5_stride0: tl.constexpr,
    const_5_stride1: tl.constexpr,
    const_5_stride2: tl.constexpr,
    lv27_ptr,
    lv27_stride0: tl.constexpr,
    lv27_stride1: tl.constexpr,
    lv27_stride2: tl.constexpr,
    lv28_ptr,
    lv28_stride0: tl.constexpr,
    lv28_stride1: tl.constexpr,
    lv28_stride2: tl.constexpr,
    lv29_ptr,
    lv29_stride0: tl.constexpr,
    lv29_stride1: tl.constexpr,
    lv30_ptr,
    lv30_stride0: tl.constexpr,
    lv30_stride1: tl.constexpr,
    lv36_ptr,
    lv36_stride0: tl.constexpr,
    lv36_stride1: tl.constexpr,
    lv38_ptr,
    lv38_stride0: tl.constexpr,
    lv38_stride1: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    # Allocate intermediate tensors
    lv31 = tl.zeros((1, 16, BLOCK_M), dtype=tl.float32)
    lv34 = tl.zeros((1, 16, 128), dtype=tl.float32)
    lv35 = tl.zeros((1, 16, BLOCK_M), dtype=tl.float32)

    # Parallel loop j from 0 to lv38_dim1 with tile size BLOCK_J
    # Executed across grid dimension 0
    j = 0 + tl.program_id(0) * BLOCK_J
    
    # Sequential loop m from 0 to 1024 with tile size BLOCK_M
    for m in range(0, 1024, BLOCK_M):
        offset_0 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv27_stride0 + (tl.arange(0, 16))[None, :, None] * lv27_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv27_stride2
        elem_j_indices = ((j // BLOCK_J) + tl.arange(0, 1))
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_17 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        temp_0 = tl.load(lv27_ptr + offset_0, mask=mask_17, other=0.0)
        offset_1 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None] * lv29_stride0 + (tl.arange(0, 16))[None, :] * lv29_stride1
        mask_18 = (elem_j_indices < 32)[:, None]
        temp_1 = tl.load(lv29_ptr + offset_1, mask=mask_18, other=0.0)
        lv31 = (temp_0 / temp_1[:, :, None])
        offset_2 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv28_stride0 + (tl.arange(0, 16))[None, :, None] * lv28_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv28_stride2
        mask_19 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, None, :]
        temp_2 = tl.load(lv28_ptr + offset_2, mask=mask_19, other=0.0)
        offset_3 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None] * lv30_stride0 + (tl.arange(0, 16))[None, :] * lv30_stride1
        mask_20 = (elem_j_indices < 32)[:, None]
        temp_3 = tl.load(lv30_ptr + offset_3, mask=mask_20, other=0.0)
        lv35 = (temp_2 / temp_3[:, :, None])
        offset_4 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_5_stride0 + (m + tl.arange(0, BLOCK_M))[None, :, None] * const_5_stride1 + (tl.arange(0, 128))[None, None, :] * const_5_stride2
        mask_21 = (elem_j_indices < 32)[:, None, None] & (m_indices < 1024)[None, :, None]
        temp_4 = tl.load(const_5_ptr + offset_4, mask=mask_21, other=0.0)
        lv34 = (tl.dot(lv31, temp_4.to(tl.float32)) + (1 * lv34))
        offset_5 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None] * lv36_stride0 + (m + tl.arange(0, BLOCK_M))[None, :] * lv36_stride1
        mask_22 = (elem_j_indices < 32)[:, None] & (m_indices < 1024)[None, :]
        tl.store(lv36_ptr + offset_5, tl.sum(lv35, axis=1, dtype=tl.float32).to(tl.float16), mask=mask_22)
    temp_5 = tl.permute(lv34, (1, 0, 2))
    offset_6 = (tl.arange(0, 16))[:, None] * lv38_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv38_stride1
    j_indices = j + tl.arange(0, BLOCK_J)
    mask_23 = (j_indices < 4096)[None, :]
    tl.store(lv38_ptr + offset_6, tl.reshape(temp_5, (16, 128)).to(tl.float16), mask=mask_23)


# Metadata for benchmark.py
TENSOR_PARAMS = ['const_1', 'const_2', 'const_3', 'const_4', 'const_5', 'const_6', 'lv24', 'lv27', 'lv28', 'lv29', 'lv30', 'lv36', 'lv38', 'x']
FP32_TENSOR_PARAMS = ['const_6', 'lv24', 'lv27', 'lv28', 'lv29', 'lv30']
BLOCK_PARAMS = ['block_k', 'block_m']

def forward(const_1, const_2, const_3, const_4, const_5, const_6, lv24, lv27, lv28, lv29, lv30, lv36, lv38, x, block_k=16, block_m=16):
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
        const_6,
        const_6.stride(0),
        const_6.stride(1),
        const_6.stride(2),
        lv24,
        lv24.stride(0),
        lv24.stride(1),
        lv24.stride(2),
        lv27,
        lv27.stride(0),
        lv27.stride(1),
        lv27.stride(2),
        lv28,
        lv28.stride(0),
        lv28.stride(1),
        lv28.stride(2),
        lv29,
        lv29.stride(0),
        lv29.stride(1),
        lv30,
        lv30.stride(0),
        lv30.stride(1),
        x,
        x.stride(0),
        x.stride(1),
        # BLOCK_K, BLOCK_M are provided by autotune,
        BLOCK_J=128,
        # BLOCK_K is automatically set by autotune,
        # BLOCK_M is automatically set by autotune
    )

    kernel_1[((4096 - 0 + 128 - 1) // 128,)](
        const_5,
        const_5.stride(0),
        const_5.stride(1),
        const_5.stride(2),
        lv27,
        lv27.stride(0),
        lv27.stride(1),
        lv27.stride(2),
        lv28,
        lv28.stride(0),
        lv28.stride(1),
        lv28.stride(2),
        lv29,
        lv29.stride(0),
        lv29.stride(1),
        lv30,
        lv30.stride(0),
        lv30.stride(1),
        lv36,
        lv36.stride(0),
        lv36.stride(1),
        lv38,
        lv38.stride(0),
        lv38.stride(1),
        # BLOCK_M are provided by autotune,
        BLOCK_J=128,
        # BLOCK_M is automatically set by autotune
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
