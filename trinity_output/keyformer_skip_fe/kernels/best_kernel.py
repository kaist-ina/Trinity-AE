import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    K1_ptr,
    K1_stride0: tl.constexpr,
    K1_stride1: tl.constexpr,
    Q1_ptr,
    Q1_stride0: tl.constexpr,
    Q1_stride1: tl.constexpr,
    V1_ptr,
    V1_stride0: tl.constexpr,
    V1_stride1: tl.constexpr,
    WK_ptr,
    WK_stride0: tl.constexpr,
    WK_stride1: tl.constexpr,
    WQ_ptr,
    WQ_stride0: tl.constexpr,
    WQ_stride1: tl.constexpr,
    WV_ptr,
    WV_stride0: tl.constexpr,
    WV_stride1: tl.constexpr,
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Initialize kernel accumulators
    K1 = tl.zeros((16, BLOCK_N), dtype=tl.float16)
    Q1 = tl.zeros((16, BLOCK_N), dtype=tl.float16)
    V1 = tl.zeros((16, BLOCK_N), dtype=tl.float16)
    # Parallel loop n from 0 to Q1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < 4096)[None, :]
        temp_0 = tl.load(X_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_1 = (k_indices < 4096)[:, None] & (n_indices < 4096)[None, :]
        temp_1 = tl.load(WQ_ptr + offset_1, mask=mask_1, other=0.0)
        Q1 = ((1 * Q1).to(tl.float16) + tl.dot(temp_0, temp_1).to(tl.float16)).to(tl.float16)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * WK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WK_stride1
        mask_2 = (k_indices < 4096)[:, None] & (n_indices < 4096)[None, :]
        temp_2 = tl.load(WK_ptr + offset_2, mask=mask_2, other=0.0)
        K1 = ((1 * K1).to(tl.float16) + tl.dot(temp_0, temp_2).to(tl.float16)).to(tl.float16)
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * WV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WV_stride1
        mask_3 = (k_indices < 4096)[:, None] & (n_indices < 4096)[None, :]
        temp_3 = tl.load(WV_ptr + offset_3, mask=mask_3, other=0.0)
        V1 = ((1 * V1).to(tl.float16) + tl.dot(temp_0, temp_3).to(tl.float16)).to(tl.float16)
    # Store kernel accumulators
    offset_4 = (tl.arange(0, 16))[:, None] * K1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * K1_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_4 = (n_indices < 4096)[None, :]
    tl.store(K1_ptr + offset_4, K1, mask=mask_4)
    offset_5 = (tl.arange(0, 16))[:, None] * Q1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * Q1_stride1
    mask_5 = (n_indices < 4096)[None, :]
    tl.store(Q1_ptr + offset_5, Q1, mask=mask_5)
    offset_6 = (tl.arange(0, 16))[:, None] * V1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * V1_stride1
    mask_6 = (n_indices < 4096)[None, :]
    tl.store(V1_ptr + offset_6, V1, mask=mask_6)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_P': 32}),
        triton.Config({'BLOCK_P': 64}),
        triton.Config({'BLOCK_P': 128})
    ], key=[]
)
@triton.jit
def kernel_1(
    C_exp_ptr,
    C_exp_stride0: tl.constexpr,
    C_exp_stride1: tl.constexpr,
    C_exp_stride2: tl.constexpr,
    C_exp_perturb_ptr,
    C_exp_perturb_stride0: tl.constexpr,
    C_exp_perturb_stride1: tl.constexpr,
    C_exp_perturb_stride2: tl.constexpr,
    C_out_ptr,
    C_out_stride0: tl.constexpr,
    C_out_stride1: tl.constexpr,
    K1_ptr,
    K1_stride0: tl.constexpr,
    K1_stride1: tl.constexpr,
    K_cache_ptr,
    K_cache_stride0: tl.constexpr,
    K_cache_stride1: tl.constexpr,
    K_cache_stride2: tl.constexpr,
    O2_ptr,
    O2_stride0: tl.constexpr,
    O2_stride1: tl.constexpr,
    Q1_ptr,
    Q1_stride0: tl.constexpr,
    Q1_stride1: tl.constexpr,
    V1_ptr,
    V1_stride0: tl.constexpr,
    V1_stride1: tl.constexpr,
    V_cache_ptr,
    V_cache_stride0: tl.constexpr,
    V_cache_stride1: tl.constexpr,
    V_cache_stride2: tl.constexpr,
    noise_ptr,
    noise_stride0: tl.constexpr,
    noise_stride1: tl.constexpr,
    noise_stride2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr
):
    # Allocate intermediate tensors
    C = tl.zeros((1, 16, BLOCK_P), dtype=tl.float16)
    C_perturb = tl.zeros((1, 16, BLOCK_P), dtype=tl.float16)
    C_sum = tl.zeros((1, 16), dtype=tl.float32)
    C_sum_perturb = tl.zeros((1, 16), dtype=tl.float32)
    K = tl.zeros((1, 16, 128), dtype=tl.float16)
    O = tl.zeros((1, 16, 128), dtype=tl.float32)
    Q = tl.zeros((1, 16, 128), dtype=tl.float16)
    V = tl.zeros((1, 16, 128), dtype=tl.float16)

    # Parallel loop n from 0 to Q1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    offset_0 = (tl.arange(0, 16))[:, None] * Q1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * Q1_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_7 = (n_indices < 4096)[None, :]
    temp_0 = tl.load(Q1_ptr + offset_0, mask=mask_7, other=0.0)
    Q2 = tl.expand_dims(temp_0, 1)
    offset_1 = (tl.arange(0, 16))[:, None] * K1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * K1_stride1
    mask_8 = (n_indices < 4096)[None, :]
    temp_1 = tl.load(K1_ptr + offset_1, mask=mask_8, other=0.0)
    K2 = tl.expand_dims(temp_1, 1)
    offset_2 = (tl.arange(0, 16))[:, None] * V1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * V1_stride1
    mask_9 = (n_indices < 4096)[None, :]
    temp_2 = tl.load(V1_ptr + offset_2, mask=mask_9, other=0.0)
    V2 = tl.expand_dims(temp_2, 1)
    # Sequential loop p from 0 to 1024 with tile size BLOCK_P
    for p in range(0, 1024, BLOCK_P):
        Q = tl.permute(Q2, (1, 0, 2))
        K = tl.permute(K2, (1, 0, 2))
        V = tl.permute(V2, (1, 0, 2))
        offset_3 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_cache_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * K_cache_stride1 + (tl.arange(0, 128))[None, None, :] * K_cache_stride2
        tl.store(K_cache_ptr + offset_3, K)
        offset_4 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_cache_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * V_cache_stride1 + (tl.arange(0, 128))[None, None, :] * V_cache_stride2
        tl.store(V_cache_ptr + offset_4, V)
        offset_5 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_cache_stride0 + (p + tl.arange(0, BLOCK_P))[None, :, None] * K_cache_stride1 + (tl.arange(0, 128))[None, None, :] * K_cache_stride2
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_10 = (p_indices < 1024)[None, :, None]
        temp_3 = tl.load(K_cache_ptr + offset_5, mask=mask_10, other=0.0)
        temp_4 = tl.permute(temp_3, (0, 2, 1))
        C = tl.dot(Q, temp_4).to(tl.float16)
        temp_5 = tl.permute(temp_3, (0, 2, 1))
        offset_6 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * noise_stride0 + (tl.arange(0, 16))[None, :, None] * noise_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * noise_stride2
        mask_11 = (p_indices < 1024)[None, None, :]
        temp_6 = tl.load(noise_ptr + offset_6, mask=mask_11, other=0.0)
        C_perturb = ((tl.dot(Q, temp_5).to(tl.float16) + temp_6).to(tl.float16) / 1.5).to(tl.float16)
        offset_7 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_exp_stride2
        mask_12 = (p_indices < 1024)[None, None, :]
        tl.store(C_exp_ptr + offset_7, tl.exp(C.to(tl.float32)), mask=mask_12)
        offset_8 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_perturb_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_perturb_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_exp_perturb_stride2
        mask_13 = (p_indices < 1024)[None, None, :]
        tl.store(C_exp_perturb_ptr + offset_8, tl.exp(C_perturb.to(tl.float32)), mask=mask_13)
        C_sum = (tl.sum(tl.exp(C.to(tl.float32)), axis=2, dtype=tl.float32) + (C_sum * 1))
        C_sum_perturb = (tl.sum(tl.exp(C_perturb.to(tl.float32)), axis=2, dtype=tl.float32) + (C_sum_perturb * 1))
    C_sum_perturb = C_sum_perturb + 0.0
    # Sequential loop p from 0 to 1024 with tile size BLOCK_P
    for p in range(0, 1024, BLOCK_P):
        offset_9 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_exp_stride2
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_14 = (p_indices < 1024)[None, None, :]
        temp_7 = tl.load(C_exp_ptr + offset_9, mask=mask_14, other=0.0)
        offset_10 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_cache_stride0 + (p + tl.arange(0, BLOCK_P))[None, :, None] * V_cache_stride1 + (tl.arange(0, 128))[None, None, :] * V_cache_stride2
        mask_15 = (p_indices < 1024)[None, :, None]
        temp_8 = tl.load(V_cache_ptr + offset_10, mask=mask_15, other=0.0)
        O = (tl.dot(temp_7, temp_8.to(tl.float32)) + (O * 1))
        offset_11 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_perturb_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_perturb_stride1 + (p + tl.arange(0, BLOCK_P))[None, None, :] * C_exp_perturb_stride2
        mask_16 = (p_indices < 1024)[None, None, :]
        temp_9 = tl.load(C_exp_perturb_ptr + offset_11, mask=mask_16, other=0.0)
        offset_12 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None] * C_out_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * C_out_stride1
        mask_17 = (p_indices < 1024)[None, :]
        tl.store(C_out_ptr + offset_12, tl.sum((temp_9 / C_sum_perturb[:, :, None]), axis=1, dtype=tl.float32).to(tl.float16), mask=mask_17)
    O = (O / C_sum[:, :, None])
    temp_10 = tl.permute(O, (1, 0, 2))
    offset_13 = (tl.arange(0, 16))[:, None] * O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O2_stride1
    mask_18 = (n_indices < 4096)[None, :]
    tl.store(O2_ptr + offset_13, tl.reshape(temp_10, (16, 128)).to(tl.float16), mask=mask_18)


# Metadata for benchmark.py
TENSOR_PARAMS = ['C_exp', 'C_exp_perturb', 'C_out', 'K1', 'K_cache', 'O2', 'Q1', 'V1', 'V_cache', 'WK', 'WQ', 'WV', 'X', 'noise']
FP32_TENSOR_PARAMS = ['C_exp', 'C_exp_perturb']
BLOCK_PARAMS = ['block_k', 'block_n', 'block_p']

def forward(C_exp, C_exp_perturb, C_out, K1, K_cache, O2, Q1, V1, V_cache, WK, WQ, WV, X, noise, block_k=16, block_n=16, block_p=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[lambda meta: ((4096 - 0 + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)](
        K1,
        K1.stride(0),
        K1.stride(1),
        Q1,
        Q1.stride(0),
        Q1.stride(1),
        V1,
        V1.stride(0),
        V1.stride(1),
        WK,
        WK.stride(0),
        WK.stride(1),
        WQ,
        WQ.stride(0),
        WQ.stride(1),
        WV,
        WV.stride(0),
        WV.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        # BLOCK_K, BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        # BLOCK_K is automatically set by autotune
    )

    kernel_1[((4096 - 0 + 128 - 1) // 128,)](
        C_exp,
        C_exp.stride(0),
        C_exp.stride(1),
        C_exp.stride(2),
        C_exp_perturb,
        C_exp_perturb.stride(0),
        C_exp_perturb.stride(1),
        C_exp_perturb.stride(2),
        C_out,
        C_out.stride(0),
        C_out.stride(1),
        K1,
        K1.stride(0),
        K1.stride(1),
        K_cache,
        K_cache.stride(0),
        K_cache.stride(1),
        K_cache.stride(2),
        O2,
        O2.stride(0),
        O2.stride(1),
        Q1,
        Q1.stride(0),
        Q1.stride(1),
        V1,
        V1.stride(0),
        V1.stride(1),
        V_cache,
        V_cache.stride(0),
        V_cache.stride(1),
        V_cache.stride(2),
        noise,
        noise.stride(0),
        noise.stride(1),
        noise.stride(2),
        # BLOCK_P are provided by autotune,
        BLOCK_N=128,
        # BLOCK_P is automatically set by autotune
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
