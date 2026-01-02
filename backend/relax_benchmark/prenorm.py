import torch
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R


def prepare_inputs_prenorm(X, WQ, WK, WV, K_cache, V_cache):
    """Prepare inputs for Relax prenorm module."""
    W_qkv = torch.cat([WQ, WK, WV], dim=1)
    return [X, W_qkv, K_cache, V_cache]


def create_relax_prenorm(M: int, N: int, H: int, D: int, P: int):
    """
    Factory function that creates a TVMScript module with static shapes.

    Args:
        M: Sequence length (new tokens)
        N: Hidden dimension
        H: Number of heads
        D: Head dimension
        P: Past sequence length (in cache)

    Returns:
        IRModule with static shapes baked in
    """
    L = P + M           # Total cache length

    @I.ir_module
    class Relax_Prenorm:

        @T.prim_func
        def update_cache(
            cache_handle: T.handle,
            new_kv_handle: T.handle,
            start_pos: T.int64
        ):
            T.func_attr({"tir.noalias": T.bool(False)})

            cache = T.match_buffer(cache_handle, (H, L, D), dtype="float16")
            new_kv = T.match_buffer(new_kv_handle, (H, M, D), dtype="float16")

            for h, m, d in T.grid(H, M, D):
                with T.block("cache_update"):
                    vh, vm, vd = T.axis.remap("SSS", [h, m, d])
                    T.reads(new_kv[vh, vm, vd])
                    T.writes(cache[vh, start_pos + vm, vd])
                    cache[vh, start_pos + vm, vd] = new_kv[vh, vm, vd]

        @R.function
        def forward(
            X: R.Tensor((M, N), "float16"),
            W_qkv: R.Tensor((N, 3 * N), "float16"),
            cache_K: R.Tensor((H, L, D), "float16"),
            cache_V: R.Tensor((H, L, D), "float16"),
        ) -> R.Tensor((M, N), dtype="float16"):

            cls = Relax_Prenorm
            with R.dataflow():
                # Pre-normalization (RMSNorm style)
                X_squared = R.multiply(X, X)
                X_variance = R.divide(R.sum(X_squared, axis=-1, keepdims=True), R.const(N, "float16"))
                X_norm = R.divide(X, R.sqrt(X_variance))

                # QKV Projection
                qkv = R.matmul(X_norm, W_qkv)
                qkv_split = R.split(qkv, indices_or_sections=3, axis=-1)

                # Reshape to [M, H, D]
                q = R.reshape(qkv_split[0], (M, H, D))
                k = R.reshape(qkv_split[1], (M, H, D))
                v = R.reshape(qkv_split[2], (M, H, D))

                # Permute to [H, M, D]
                q = R.permute_dims(q, axes=[1, 0, 2])
                k = R.permute_dims(k, axes=[1, 0, 2])
                v = R.permute_dims(v, axes=[1, 0, 2])

                # KV Cache update: in-place update at position P
                cache_K = R.call_tir_inplace(
                    cls.update_cache,
                    (cache_K, k, T.int64(P)),
                    inplace_indices=[0],
                    out_sinfo=R.Tensor((H, L, D), dtype="float16")
                )
                cache_V = R.call_tir_inplace(
                    cls.update_cache,
                    (cache_V, v, T.int64(P)),
                    inplace_indices=[0],
                    out_sinfo=R.Tensor((H, L, D), dtype="float16")
                )

                # Attention: Q @ K^T -> softmax -> @ V
                k_for_attn = R.permute_dims(cache_K, axes=[0, 2, 1])  # [H, D, L]
                scores = R.matmul(q, k_for_attn)  # [H, M, L]

                # Softmax
                weights = R.nn.softmax(scores, axis=-1)

                # Weighted sum of values
                out = R.matmul(weights, cache_V)  # [H, M, D]

                # Permute back to [M, H, D] and reshape to [M, N]
                out = R.permute_dims(out, axes=[1, 0, 2])
                out = R.reshape(out, (M, N))
                R.output(out)

            return out

    return Relax_Prenorm
