import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_J': 32, 'BLOCK_I': 32}),
        triton.Config({'BLOCK_J': 32, 'BLOCK_I': 64}),
        triton.Config({'BLOCK_J': 32, 'BLOCK_I': 128}),
        triton.Config({'BLOCK_J': 64, 'BLOCK_I': 32}),
        triton.Config({'BLOCK_J': 64, 'BLOCK_I': 64}),
        triton.Config({'BLOCK_J': 64, 'BLOCK_I': 128}),
        triton.Config({'BLOCK_J': 128, 'BLOCK_I': 32}),
        triton.Config({'BLOCK_J': 128, 'BLOCK_I': 64}),
        triton.Config({'BLOCK_J': 128, 'BLOCK_I': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    lv1_ptr,
    lv1_stride0: tl.constexpr,
    lv1_stride1: tl.constexpr,
    lv3_ptr,
    lv3_stride0: tl.constexpr,
    lv3_stride1: tl.constexpr,
    lv5_ptr,
    lv5_stride0: tl.constexpr,
    lv5_stride1: tl.constexpr,
    p_k_proj_weight_ptr,
    p_k_proj_weight_stride0: tl.constexpr,
    p_k_proj_weight_stride1: tl.constexpr,
    p_q_proj_weight_ptr,
    p_q_proj_weight_stride0: tl.constexpr,
    p_q_proj_weight_stride1: tl.constexpr,
    p_v_proj_weight_ptr,
    p_v_proj_weight_stride0: tl.constexpr,
    p_v_proj_weight_stride1: tl.constexpr,
    x_ptr,
    x_stride0: tl.constexpr,
    x_stride1: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_I: tl.constexpr
):
    # Initialize kernel accumulators
    lv1 = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    lv3 = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    lv5 = tl.zeros((16, BLOCK_J), dtype=tl.float16)
    # Parallel loop j from 0 to lv1_dim1 with tile size BLOCK_J
    # Executed across grid dimension 0
    j = 0 + tl.program_id(0) * BLOCK_J
    
    # Sequential loop i from 0 to 4096 with tile size BLOCK_I
    for i in range(0, 4096, BLOCK_I):
        offset_0 = (tl.arange(0, 16))[:, None] * x_stride0 + (i + tl.arange(0, BLOCK_I))[None, :] * x_stride1
        i_indices = i + tl.arange(0, BLOCK_I)
        mask_0 = (i_indices < 4096)[None, :]
        temp_0 = tl.load(x_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (j + tl.arange(0, BLOCK_J))[:, None] * p_q_proj_weight_stride0 + (i + tl.arange(0, BLOCK_I))[None, :] * p_q_proj_weight_stride1
        j_indices = j + tl.arange(0, BLOCK_J)
        mask_1 = (j_indices < 4096)[:, None] & (i_indices < 4096)[None, :]
        temp_1 = tl.load(p_q_proj_weight_ptr + offset_1, mask=mask_1, other=0.0)
        temp_2 = tl.trans(temp_1)
        lv1 = (tl.dot(temp_0, temp_2).to(tl.float16) + (lv1 * 1).to(tl.float16)).to(tl.float16)
        offset_2 = (j + tl.arange(0, BLOCK_J))[:, None] * p_k_proj_weight_stride0 + (i + tl.arange(0, BLOCK_I))[None, :] * p_k_proj_weight_stride1
        mask_2 = (j_indices < 4096)[:, None] & (i_indices < 4096)[None, :]
        temp_3 = tl.load(p_k_proj_weight_ptr + offset_2, mask=mask_2, other=0.0)
        temp_4 = tl.trans(temp_3)
        lv3 = (tl.dot(temp_0, temp_4).to(tl.float16) + (lv3 * 1).to(tl.float16)).to(tl.float16)
        offset_3 = (j + tl.arange(0, BLOCK_J))[:, None] * p_v_proj_weight_stride0 + (i + tl.arange(0, BLOCK_I))[None, :] * p_v_proj_weight_stride1
        mask_3 = (j_indices < 4096)[:, None] & (i_indices < 4096)[None, :]
        temp_5 = tl.load(p_v_proj_weight_ptr + offset_3, mask=mask_3, other=0.0)
        temp_6 = tl.trans(temp_5)
        lv5 = (tl.dot(temp_0, temp_6).to(tl.float16) + (lv5 * 1).to(tl.float16)).to(tl.float16)
    # Store kernel accumulators
    offset_4 = (tl.arange(0, 16))[:, None] * lv1_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv1_stride1
    j_indices = j + tl.arange(0, BLOCK_J)
    mask_4 = (j_indices < 4096)[None, :]
    tl.store(lv1_ptr + offset_4, lv1, mask=mask_4)
    offset_5 = (tl.arange(0, 16))[:, None] * lv3_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv3_stride1
    mask_5 = (j_indices < 4096)[None, :]
    tl.store(lv3_ptr + offset_5, lv3, mask=mask_5)
    offset_6 = (tl.arange(0, 16))[:, None] * lv5_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv5_stride1
    mask_6 = (j_indices < 4096)[None, :]
    tl.store(lv5_ptr + offset_6, lv5, mask=mask_6)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_M': 32}),
        triton.Config({'BLOCK_M': 64}),
        triton.Config({'BLOCK_M': 128})
    ], key=[]
)
@triton.jit
def kernel_1(
    const_1_ptr,
    const_1_stride0: tl.constexpr,
    const_1_stride1: tl.constexpr,
    const_1_stride2: tl.constexpr,
    const_2_ptr,
    const_2_stride0: tl.constexpr,
    const_2_stride1: tl.constexpr,
    const_2_stride2: tl.constexpr,
    lv1_ptr,
    lv1_stride0: tl.constexpr,
    lv1_stride1: tl.constexpr,
    lv28_ptr,
    lv28_stride0: tl.constexpr,
    lv28_stride1: tl.constexpr,
    lv28_stride2: tl.constexpr,
    lv3_ptr,
    lv3_stride0: tl.constexpr,
    lv3_stride1: tl.constexpr,
    lv35_ptr,
    lv35_stride0: tl.constexpr,
    lv35_stride1: tl.constexpr,
    lv37_ptr,
    lv37_stride0: tl.constexpr,
    lv37_stride1: tl.constexpr,
    lv39_ptr,
    lv39_stride0: tl.constexpr,
    lv39_stride1: tl.constexpr,
    lv5_ptr,
    lv5_stride0: tl.constexpr,
    lv5_stride1: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    # Allocate intermediate tensors
    lv29 = tl.zeros((1, 16), dtype=tl.float32)
    lv31 = tl.zeros((1, 16, BLOCK_M), dtype=tl.float32)
    lv34 = tl.zeros((1, 16, 128), dtype=tl.float32)

    # Parallel loop j from 0 to lv1_dim1 with tile size BLOCK_J
    # Executed across grid dimension 0
    j = 0 + tl.program_id(0) * BLOCK_J
    
    offset_0 = (tl.arange(0, 16))[:, None] * lv1_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv1_stride1
    j_indices = j + tl.arange(0, BLOCK_J)
    mask_7 = (j_indices < 4096)[None, :]
    temp_0 = tl.load(lv1_ptr + offset_0, mask=mask_7, other=0.0)
    temp_1 = tl.expand_dims(temp_0, 1)
    lv9 = tl.permute(temp_1, (1, 0, 2))
    offset_1 = (tl.arange(0, 16))[:, None] * lv3_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv3_stride1
    mask_8 = (j_indices < 4096)[None, :]
    temp_2 = tl.load(lv3_ptr + offset_1, mask=mask_8, other=0.0)
    temp_3 = tl.expand_dims(temp_2, 1)
    lv10 = tl.permute(temp_3, (1, 0, 2))
    offset_2 = (tl.arange(0, 16))[:, None] * lv5_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv5_stride1
    mask_9 = (j_indices < 4096)[None, :]
    temp_4 = tl.load(lv5_ptr + offset_2, mask=mask_9, other=0.0)
    temp_5 = tl.expand_dims(temp_4, 1)
    lv11 = tl.permute(temp_5, (1, 0, 2))
    offset_3 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_1_stride0 + (1024 + tl.arange(0, 16))[None, :, None] * const_1_stride1 + (tl.arange(0, 128))[None, None, :] * const_1_stride2
    tl.store(const_1_ptr + offset_3, lv10)
    offset_4 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_2_stride0 + (1024 + tl.arange(0, 16))[None, :, None] * const_2_stride1 + (tl.arange(0, 128))[None, None, :] * const_2_stride2
    tl.store(const_2_ptr + offset_4, lv11)
    # Sequential loop m from 0 to 1040 with tile size BLOCK_M
    for m in range(0, 1040, BLOCK_M):
        offset_5 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_1_stride0 + (m + tl.arange(0, BLOCK_M))[None, :, None] * const_1_stride1 + (tl.arange(0, 128))[None, None, :] * const_1_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_10 = (m_indices < 1040)[None, :, None]
        temp_6 = tl.load(const_1_ptr + offset_5, mask=mask_10, other=0.0)
        temp_7 = tl.permute(temp_6, (0, 2, 1))
        offset_6 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv28_stride0 + (tl.arange(0, 16))[None, :, None] * lv28_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv28_stride2
        mask_11 = (m_indices < 1040)[None, None, :]
        tl.store(lv28_ptr + offset_6, tl.exp(tl.dot(lv9, temp_7).to(tl.float32)), mask=mask_11)
        temp_8 = tl.permute(temp_6, (0, 2, 1))
        lv29 = (tl.sum(tl.exp(tl.dot(lv9, temp_8).to(tl.float32)), axis=2, dtype=tl.float32) + (1 * lv29))
    lv29 = lv29 + 0.0
    # Sequential loop m from 0 to 1040 with tile size BLOCK_M
    for m in range(0, 1040, BLOCK_M):
        offset_7 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * lv28_stride0 + (tl.arange(0, 16))[None, :, None] * lv28_stride1 + (m + tl.arange(0, BLOCK_M))[None, None, :] * lv28_stride2
        m_indices = m + tl.arange(0, BLOCK_M)
        mask_12 = (m_indices < 1040)[None, None, :]
        temp_9 = tl.load(lv28_ptr + offset_7, mask=mask_12, other=0.0)
        lv31 = (temp_9 / lv29[:, :, None])
        offset_8 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None, None] * const_2_stride0 + (m + tl.arange(0, BLOCK_M))[None, :, None] * const_2_stride1 + (tl.arange(0, 128))[None, None, :] * const_2_stride2
        mask_13 = (m_indices < 1040)[None, :, None]
        temp_10 = tl.load(const_2_ptr + offset_8, mask=mask_13, other=0.0)
        lv34 = ((lv34 * 1) + tl.dot(lv31, temp_10.to(tl.float32)))
        offset_9 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None] * lv35_stride0 + (m + tl.arange(0, BLOCK_M))[None, :] * lv35_stride1
        mask_14 = (m_indices < 1040)[None, :]
        tl.store(lv35_ptr + offset_9, tl.sum(lv31, axis=1, dtype=tl.float32).to(tl.float16), mask=mask_14)
        offset_10 = (((j // BLOCK_J)+tl.arange(0, 1)))[:, None] * lv37_stride0 + (m + tl.arange(0, BLOCK_M))[None, :] * lv37_stride1
        mask_15 = (m_indices < 1040)[None, :]
        tl.store(lv37_ptr + offset_10, tl.sum((lv31 * lv31), axis=1, dtype=tl.float32).to(tl.float16), mask=mask_15)
    temp_11 = tl.permute(lv34, (1, 0, 2))
    offset_11 = (tl.arange(0, 16))[:, None] * lv39_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * lv39_stride1
    mask_16 = (j_indices < 4096)[None, :]
    tl.store(lv39_ptr + offset_11, tl.reshape(temp_11, (16, 128)).to(tl.float16), mask=mask_16)


# Metadata for benchmark.py
TENSOR_PARAMS = ['const_1', 'const_2', 'lv1', 'lv28', 'lv3', 'lv35', 'lv37', 'lv39', 'lv5', 'p_k_proj_weight', 'p_q_proj_weight', 'p_v_proj_weight', 'x']
FP32_TENSOR_PARAMS = ['lv28']
BLOCK_PARAMS = ['block_i', 'block_j', 'block_m']

def forward(const_1, const_2, lv1, lv28, lv3, lv35, lv37, lv39, lv5, p_k_proj_weight, p_q_proj_weight, p_v_proj_weight, x, block_i=16, block_j=16, block_m=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[lambda meta: ((4096 - 0 + meta["BLOCK_J"] - 1) // meta["BLOCK_J"],)](
        lv1,
        lv1.stride(0),
        lv1.stride(1),
        lv3,
        lv3.stride(0),
        lv3.stride(1),
        lv5,
        lv5.stride(0),
        lv5.stride(1),
        p_k_proj_weight,
        p_k_proj_weight.stride(0),
        p_k_proj_weight.stride(1),
        p_q_proj_weight,
        p_q_proj_weight.stride(0),
        p_q_proj_weight.stride(1),
        p_v_proj_weight,
        p_v_proj_weight.stride(0),
        p_v_proj_weight.stride(1),
        x,
        x.stride(0),
        x.stride(1),
        # BLOCK_I, BLOCK_J are provided by autotune,
        # BLOCK_J is automatically set by autotune,
        # BLOCK_I is automatically set by autotune
    )

    kernel_1[((4096 - 0 + 128 - 1) // 128,)](
        const_1,
        const_1.stride(0),
        const_1.stride(1),
        const_1.stride(2),
        const_2,
        const_2.stride(0),
        const_2.stride(1),
        const_2.stride(2),
        lv1,
        lv1.stride(0),
        lv1.stride(1),
        lv28,
        lv28.stride(0),
        lv28.stride(1),
        lv28.stride(2),
        lv3,
        lv3.stride(0),
        lv3.stride(1),
        lv35,
        lv35.stride(0),
        lv35.stride(1),
        lv37,
        lv37.stride(0),
        lv37.stride(1),
        lv39,
        lv39.stride(0),
        lv39.stride(1),
        lv5,
        lv5.stride(0),
        lv5.stride(1),
        # BLOCK_M are provided by autotune,
        BLOCK_J=128,
        # BLOCK_M is automatically set by autotune
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
