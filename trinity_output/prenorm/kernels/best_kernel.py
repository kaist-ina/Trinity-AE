import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32, 'BLOCK_J': 32}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_J': 64}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_J': 128}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_J': 32}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_J': 64}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_J': 128}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_J': 32}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_J': 64}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_J': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    lv11_ptr,
    lv11_stride0: tl.constexpr,
    lv11_stride1: tl.constexpr,
    lv7_ptr,
    lv7_stride0: tl.constexpr,
    lv7_stride1: tl.constexpr,
    lv9_ptr,
    lv9_stride0: tl.constexpr,
    lv9_stride1: tl.constexpr,
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
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr
):
    # Allocate intermediate tensors
    lv1 = tl.zeros((16,), dtype=tl.float32)

    # Initialize kernel accumulators
    lv11 = tl.zeros((16, BLOCK_K), dtype=tl.float32)
    lv7 = tl.zeros((16, BLOCK_K), dtype=tl.float32)
    lv9 = tl.zeros((16, BLOCK_K), dtype=tl.float32)
    # Parallel loop k from 0 to lv7_dim1 with tile size BLOCK_K
    # Executed across grid dimension 0
    k = 0 + tl.program_id(0) * BLOCK_K
    
    # Sequential loop j from 0 to 4096 with tile size BLOCK_J
    for j in range(0, 4096, BLOCK_J):
        offset_0 = (tl.arange(0, 16))[:, None] * x_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * x_stride1
        j_indices = j + tl.arange(0, BLOCK_J)
        mask_0 = (j_indices < 4096)[None, :]
        temp_0 = tl.load(x_ptr + offset_0, mask=mask_0, other=0.0)
        lv1 = (tl.sum((temp_0 * temp_0), axis=1, dtype=tl.float32) + (lv1 * 1))
    # Skipped empty sloop with dummy body
    # Sequential loop j from 0 to 4096 with tile size BLOCK_J
    for j in range(0, 4096, BLOCK_J):
        offset_1 = (tl.arange(0, 16))[:, None] * x_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * x_stride1
        j_indices = j + tl.arange(0, BLOCK_J)
        mask_1 = (j_indices < 4096)[None, :]
        temp_1 = tl.load(x_ptr + offset_1, mask=mask_1, other=0.0)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * p_q_proj_weight_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * p_q_proj_weight_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_2 = (k_indices < 4096)[:, None] & (j_indices < 4096)[None, :]
        temp_2 = tl.load(p_q_proj_weight_ptr + offset_2, mask=mask_2, other=0.0)
        temp_3 = tl.trans(temp_2)
        lv7 = (tl.dot(temp_1.to(tl.float32), temp_3.to(tl.float32)) + (1 * lv7))
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * p_k_proj_weight_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * p_k_proj_weight_stride1
        mask_3 = (k_indices < 4096)[:, None] & (j_indices < 4096)[None, :]
        temp_4 = tl.load(p_k_proj_weight_ptr + offset_3, mask=mask_3, other=0.0)
        temp_5 = tl.trans(temp_4)
        lv9 = (tl.dot(temp_1.to(tl.float32), temp_5.to(tl.float32)) + (1 * lv9))
        offset_4 = (k + tl.arange(0, BLOCK_K))[:, None] * p_v_proj_weight_stride0 + (j + tl.arange(0, BLOCK_J))[None, :] * p_v_proj_weight_stride1
        mask_4 = (k_indices < 4096)[:, None] & (j_indices < 4096)[None, :]
        temp_6 = tl.load(p_v_proj_weight_ptr + offset_4, mask=mask_4, other=0.0)
        temp_7 = tl.trans(temp_6)
        lv11 = (tl.dot(temp_1.to(tl.float32), temp_7.to(tl.float32)) + (1 * lv11))
    lv7 = (lv7 / tl.sqrt((lv1 / 4096.0).to(tl.float32))[:, None])
    lv9 = (lv9 / tl.sqrt((lv1 / 4096.0).to(tl.float32))[:, None])
    lv11 = (lv11 / tl.sqrt((lv1 / 4096.0).to(tl.float32))[:, None])
    # Store kernel accumulators
    offset_5 = (tl.arange(0, 16))[:, None] * lv11_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * lv11_stride1
    k_indices = k + tl.arange(0, BLOCK_K)
    mask_5 = (k_indices < 4096)[None, :]
    tl.store(lv11_ptr + offset_5, lv11.to(tl.float16), mask=mask_5)
    offset_6 = (tl.arange(0, 16))[:, None] * lv7_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * lv7_stride1
    mask_6 = (k_indices < 4096)[None, :]
    tl.store(lv7_ptr + offset_6, lv7.to(tl.float16), mask=mask_6)
    offset_7 = (tl.arange(0, 16))[:, None] * lv9_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * lv9_stride1
    mask_7 = (k_indices < 4096)[None, :]
    tl.store(lv9_ptr + offset_7, lv9.to(tl.float16), mask=mask_7)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_O': 32}),
        triton.Config({'BLOCK_O': 64}),
        triton.Config({'BLOCK_O': 128})
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
    lv11_ptr,
    lv11_stride0: tl.constexpr,
    lv11_stride1: tl.constexpr,
    lv32_ptr,
    lv32_stride0: tl.constexpr,
    lv32_stride1: tl.constexpr,
    lv7_ptr,
    lv7_stride0: tl.constexpr,
    lv7_stride1: tl.constexpr,
    lv9_ptr,
    lv9_stride0: tl.constexpr,
    lv9_stride1: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_O: tl.constexpr
):
    # Allocate intermediate tensors
    lv24 = tl.zeros((1, 16, BLOCK_O), dtype=tl.float32)
    lv25 = tl.zeros((1, 16), dtype=tl.float32)
    lv30 = tl.zeros((1, 16, 128), dtype=tl.float32)

    # Parallel loop l from 0 to lv7_dim1 with tile size BLOCK_L
    # Executed across grid dimension 0
    l = 0 + tl.program_id(0) * BLOCK_L
    
    offset_0 = (tl.arange(0, 16))[:, None] * lv7_stride0 + (l + tl.arange(0, BLOCK_L))[None, :] * lv7_stride1
    l_indices = l + tl.arange(0, BLOCK_L)
    mask_8 = (l_indices < 4096)[None, :]
    temp_0 = tl.load(lv7_ptr + offset_0, mask=mask_8, other=0.0)
    lv12 = tl.expand_dims(temp_0, 1)
    offset_1 = (tl.arange(0, 16))[:, None] * lv9_stride0 + (l + tl.arange(0, BLOCK_L))[None, :] * lv9_stride1
    mask_9 = (l_indices < 4096)[None, :]
    temp_1 = tl.load(lv9_ptr + offset_1, mask=mask_9, other=0.0)
    lv13 = tl.expand_dims(temp_1, 1)
    offset_2 = (tl.arange(0, 16))[:, None] * lv11_stride0 + (l + tl.arange(0, BLOCK_L))[None, :] * lv11_stride1
    mask_10 = (l_indices < 4096)[None, :]
    temp_2 = tl.load(lv11_ptr + offset_2, mask=mask_10, other=0.0)
    lv14 = tl.expand_dims(temp_2, 1)
    offset_3 = (((l // BLOCK_L)+tl.arange(0, 1)))[:, None, None] * const_1_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * const_1_stride1 + (tl.arange(0, 128))[None, None, :] * const_1_stride2
    elem_l_indices = ((l // BLOCK_L) + tl.arange(0, 1))
    mask_11 = (elem_l_indices < 32)[:, None, None]
    tl.store(const_1_ptr + offset_3, tl.permute(lv13, (1, 0, 2)).to(tl.float32), mask=mask_11)
    offset_4 = (((l // BLOCK_L)+tl.arange(0, 1)))[:, None, None] * const_2_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * const_2_stride1 + (tl.arange(0, 128))[None, None, :] * const_2_stride2
    mask_12 = (elem_l_indices < 32)[:, None, None]
    tl.store(const_2_ptr + offset_4, tl.permute(lv14, (1, 0, 2)).to(tl.float16), mask=mask_12)
    # Sequential loop o from 0 to 1024 with tile size BLOCK_O
    for o in range(0, 1024, BLOCK_O):
        temp_3 = tl.permute(lv12, (1, 0, 2))
        offset_5 = (((l // BLOCK_L)+tl.arange(0, 1)))[:, None, None] * const_1_stride0 + (o + tl.arange(0, BLOCK_O))[None, :, None] * const_1_stride1 + (tl.arange(0, 128))[None, None, :] * const_1_stride2
        elem_l_indices = ((l // BLOCK_L) + tl.arange(0, 1))
        o_indices = o + tl.arange(0, BLOCK_O)
        mask_13 = (elem_l_indices < 32)[:, None, None] & (o_indices < 1024)[None, :, None]
        temp_4 = tl.load(const_1_ptr + offset_5, mask=mask_13, other=0.0)
        temp_5 = tl.permute(temp_4, (0, 2, 1))
        lv24 = tl.exp(tl.dot(temp_3, temp_5).to(tl.float32))
        lv25 = ((lv25 * 1) + tl.sum(lv24, axis=2, dtype=tl.float32))
        offset_6 = (((l // BLOCK_L)+tl.arange(0, 1)))[:, None, None] * const_2_stride0 + (o + tl.arange(0, BLOCK_O))[None, :, None] * const_2_stride1 + (tl.arange(0, 128))[None, None, :] * const_2_stride2
        mask_14 = (elem_l_indices < 32)[:, None, None] & (o_indices < 1024)[None, :, None]
        temp_6 = tl.load(const_2_ptr + offset_6, mask=mask_14, other=0.0)
        lv30 = (tl.dot(lv24, temp_6.to(tl.float32)) + (lv30 * 1))
    # Skipped empty sloop with dummy body
    lv30 = (lv30 / lv25[:, :, None])
    temp_7 = tl.permute(lv30, (1, 0, 2))
    offset_7 = (tl.arange(0, 16))[:, None] * lv32_stride0 + (l + tl.arange(0, BLOCK_L))[None, :] * lv32_stride1
    mask_15 = (l_indices < 4096)[None, :]
    tl.store(lv32_ptr + offset_7, tl.reshape(temp_7, (16, 128)).to(tl.float16), mask=mask_15)


# Metadata for benchmark.py
TENSOR_PARAMS = ['const_1', 'const_2', 'lv11', 'lv32', 'lv7', 'lv9', 'p_k_proj_weight', 'p_q_proj_weight', 'p_v_proj_weight', 'x']
FP32_TENSOR_PARAMS = ['const_1', 'lv11', 'lv7', 'lv9']
BLOCK_PARAMS = ['block_j', 'block_k', 'block_o']

def forward(const_1, const_2, lv11, lv32, lv7, lv9, p_k_proj_weight, p_q_proj_weight, p_v_proj_weight, x, block_j=16, block_k=16, block_o=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[lambda meta: ((4096 - 0 + meta["BLOCK_K"] - 1) // meta["BLOCK_K"],)](
        lv11,
        lv11.stride(0),
        lv11.stride(1),
        lv7,
        lv7.stride(0),
        lv7.stride(1),
        lv9,
        lv9.stride(0),
        lv9.stride(1),
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
        # BLOCK_J, BLOCK_K are provided by autotune,
        # BLOCK_K is automatically set by autotune,
        # BLOCK_J is automatically set by autotune
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
        lv11,
        lv11.stride(0),
        lv11.stride(1),
        lv32,
        lv32.stride(0),
        lv32.stride(1),
        lv7,
        lv7.stride(0),
        lv7.stride(1),
        lv9,
        lv9.stride(0),
        lv9.stride(1),
        # BLOCK_O are provided by autotune,
        BLOCK_L=128,
        # BLOCK_O is automatically set by autotune
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
