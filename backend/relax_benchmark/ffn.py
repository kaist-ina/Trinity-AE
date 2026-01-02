from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R


def prepare_inputs_ffn(X, O2, WO, WFF1a, WFF1b, WFF2):
    """Prepare inputs for Relax FFN module."""
    return [X, O2, WO, WFF1a, WFF1b, WFF2]


def create_relax_ffn(M: int, N: int, N4: int):
    """
    Factory function that creates a Relax IR module for FFN (Feed-Forward Network).

    This implements the following PyTorch operations:
        attn_O1 = O2 @ WO
        attn_O2 = attn_O1 + X
        attn_O3 = attn_O2.pow(2).mean(-1, keepdim=True)
        attn_O_norm = attn_O2 * rsqrt(attn_O3)
        FF1a = attn_O_norm @ WFF1a
        FF1b = attn_O_norm @ WFF1b
        FF1b_silu = FF1b * sigmoid(FF1b)
        FF1 = FF1a * FF1b_silu
        FF2 = FF1 @ WFF2

    Args:
        M: Sequence length
        N: Hidden dimension
        N4: Intermediate FFN dimension (typically 4 * N)

    Returns:
        IRModule with static shapes baked in
    """

    @I.ir_module
    class Relax_FFN:

        @T.prim_func(private=True)
        def _shape_anchor(placeholder: T.handle):
            T.func_attr({"tir.noalias": T.bool(True), "tir.is_scheduled": T.bool(True)})
            buf = T.match_buffer(placeholder, (M, N, N4), dtype="float16")
            T.evaluate(0)

        @R.function
        def forward(
            X: R.Tensor((M, N), "float16"),
            O2: R.Tensor((M, N), "float16"),
            WO: R.Tensor((N, N), "float16"),
            WFF1a: R.Tensor((N, N4), "float16"),
            WFF1b: R.Tensor((N, N4), "float16"),
            WFF2: R.Tensor((N4, N), "float16")
        ) -> R.Tensor((M, N), dtype="float16"):
            with R.dataflow():
                # Attention output projection and residual connection
                attn_O1 = R.matmul(O2, WO)
                attn_O2 = R.add(attn_O1, X)

                # RMSNorm: mean of squared values
                attn_O2_squared = R.multiply(attn_O2, attn_O2)
                attn_O3 = R.mean(attn_O2_squared, axis=-1, keepdims=True)
                attn_O_norm = R.multiply(attn_O2, R.rsqrt(attn_O3))

                # SwiGLU: gate * silu(up)
                FF1a = R.matmul(attn_O_norm, WFF1a)
                FF1b = R.matmul(attn_O_norm, WFF1b)
                FF1b_silu = R.nn.silu(FF1b)
                FF1 = R.multiply(FF1a, FF1b_silu)

                # Down projection
                FF2 = R.matmul(FF1, WFF2)

                R.output(FF2)
            return FF2

    return Relax_FFN
