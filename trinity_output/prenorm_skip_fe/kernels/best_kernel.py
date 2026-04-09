import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_K': 32, 'BLOCK_P': 128}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_K': 64, 'BLOCK_P': 128}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_K': 128, 'BLOCK_P': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    K_cache_ptr,
    K_cache_stride0: tl.constexpr,
    K_cache_stride1: tl.constexpr,
    K_cache_stride2: tl.constexpr,
    O2_ptr,
    O2_stride0: tl.constexpr,
    O2_stride1: tl.constexpr,
    V_cache_ptr,
    V_cache_stride0: tl.constexpr,
    V_cache_stride1: tl.constexpr,
    V_cache_stride2: tl.constexpr,
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
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr
):
    # Allocate intermediate tensors
    C_exp = tl.zeros((1, 16, BLOCK_P), dtype=tl.float32)
    C_sum = tl.zeros((1, 16), dtype=tl.float32)
    K1 = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    O = tl.zeros((1, 16, 128), dtype=tl.float32)
    Q1 = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    V1 = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    X2 = tl.zeros((16,), dtype=tl.float16)

    # Parallel loop n from 0 to Q1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < 4096)[None, :]
        temp_0 = tl.load(X_ptr + offset_0, mask=mask_0, other=0.0)
        X2 = ((X2 * 1).to(tl.float16) + tl.sum((temp_0 * temp_0).to(tl.float16), axis=1, dtype=tl.float16)).to(tl.float16)
    # Skipped empty sloop with dummy body
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_1 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_1 = (k_indices < 4096)[None, :]
        temp_1 = tl.load(X_ptr + offset_1, mask=mask_1, other=0.0)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * WQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_2 = (k_indices < 4096)[:, None] & (n_indices < 4096)[None, :]
        temp_2 = tl.load(WQ_ptr + offset_2, mask=mask_2, other=0.0)
        Q1 = ((1 * Q1) + tl.dot(temp_1.to(tl.float32), temp_2.to(tl.float32)))
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * WK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WK_stride1
        mask_3 = (k_indices < 4096)[:, None] & (n_indices < 4096)[None, :]
        temp_3 = tl.load(WK_ptr + offset_3, mask=mask_3, other=0.0)
        K1 = ((1 * K1) + tl.dot(temp_1.to(tl.float32), temp_3.to(tl.float32)))
        offset_4 = (k + tl.arange(0, BLOCK_K))[:, None] * WV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WV_stride1
        mask_4 = (k_indices < 4096)[:, None] & (n_indices < 4096)[None, :]
        temp_4 = tl.load(WV_ptr + offset_4, mask=mask_4, other=0.0)
        V1 = ((1 * V1) + tl.dot(temp_1.to(tl.float32), temp_4.to(tl.float32)))
    Q1 = (Q1 / tl.sqrt((X2 / 4096).to(tl.float32))[:, None])
    K1 = (K1 / tl.sqrt((X2 / 4096).to(tl.float32))[:, None])
    V1 = (V1 / tl.sqrt((X2 / 4096).to(tl.float32))[:, None])
    temp_5 = tl.expand_dims(Q1, 1)
    Q = tl.permute(temp_5, (1, 0, 2))
    temp_6 = tl.expand_dims(K1, 1)
    K = tl.permute(temp_6, (1, 0, 2))
    temp_7 = tl.expand_dims(V1, 1)
    V = tl.permute(temp_7, (1, 0, 2))
    offset_5 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_cache_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * K_cache_stride1 + (tl.arange(0, 128))[None, None, :] * K_cache_stride2
    elem_n_indices = ((n // BLOCK_N) + tl.arange(0, 1))
    mask_5 = (elem_n_indices < 32)[:, None, None]
    tl.store(K_cache_ptr + offset_5, K.to(tl.float16), mask=mask_5)
    offset_6 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_cache_stride0 + (1008 + tl.arange(0, 16))[None, :, None] * V_cache_stride1 + (tl.arange(0, 128))[None, None, :] * V_cache_stride2
    mask_6 = (elem_n_indices < 32)[:, None, None]
    tl.store(V_cache_ptr + offset_6, V.to(tl.float16), mask=mask_6)
    # Sequential loop p from 0 to 1024 with tile size BLOCK_P
    for p in range(0, 1024, BLOCK_P):
        offset_7 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_cache_stride0 + (p + tl.arange(0, BLOCK_P))[None, :, None] * K_cache_stride1 + (tl.arange(0, 128))[None, None, :] * K_cache_stride2
        elem_n_indices = ((n // BLOCK_N) + tl.arange(0, 1))
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_7 = (elem_n_indices < 32)[:, None, None] & (p_indices < 1024)[None, :, None]
        temp_8 = tl.load(K_cache_ptr + offset_7, mask=mask_7, other=0.0)
        temp_9 = tl.permute(temp_8, (0, 2, 1))
        C_exp = tl.exp(tl.dot(Q, temp_9.to(tl.float32)).to(tl.float32))
        C_sum = ((C_sum * 1) + tl.sum(C_exp, axis=2, dtype=tl.float32))
        offset_8 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_cache_stride0 + (p + tl.arange(0, BLOCK_P))[None, :, None] * V_cache_stride1 + (tl.arange(0, 128))[None, None, :] * V_cache_stride2
        mask_8 = (elem_n_indices < 32)[:, None, None] & (p_indices < 1024)[None, :, None]
        temp_10 = tl.load(V_cache_ptr + offset_8, mask=mask_8, other=0.0)
        O = (tl.dot(C_exp, temp_10.to(tl.float32)) + (1 * O))
    # Skipped empty sloop with dummy body
    O = (O / C_sum[:, :, None])
    temp_11 = tl.permute(O, (1, 0, 2))
    offset_9 = (tl.arange(0, 16))[:, None] * O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O2_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_9 = (n_indices < 4096)[None, :]
    tl.store(O2_ptr + offset_9, tl.reshape(temp_11, (16, 128)).to(tl.float16), mask=mask_9)


# Metadata for benchmark.py
TENSOR_PARAMS = ['K_cache', 'O2', 'V_cache', 'WK', 'WQ', 'WV', 'X']
FP32_TENSOR_PARAMS = []
BLOCK_PARAMS = ['block_k', 'block_p']

def forward(K_cache, O2, V_cache, WK, WQ, WV, X, block_k=16, block_p=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4096 - 0 + 128 - 1) // 128,)](
        K_cache,
        K_cache.stride(0),
        K_cache.stride(1),
        K_cache.stride(2),
        O2,
        O2.stride(0),
        O2.stride(1),
        V_cache,
        V_cache.stride(0),
        V_cache.stride(1),
        V_cache.stride(2),
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
        # BLOCK_K, BLOCK_P are provided by autotune,
        BLOCK_N=128,
        # BLOCK_K is automatically set by autotune,
        # BLOCK_P is automatically set by autotune
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
