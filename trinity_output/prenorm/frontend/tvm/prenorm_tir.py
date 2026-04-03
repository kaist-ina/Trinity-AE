# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def broadcast_to(lv15: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(128)):
            with T.sblock("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv15[v_ax0, v_ax1, v_ax2])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                T_broadcast_to[v_ax0, v_ax1, v_ax2] = lv15[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def broadcast_to1(lv20: T.Buffer((T.int64(32), T.int64(128), T.int64(1024)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(128), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(128), T.int64(1024)):
            with T.sblock("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv20[v_ax0, v_ax1, v_ax2])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                T_broadcast_to[v_ax0, v_ax1, v_ax2] = lv20[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def broadcast_to2(lv27: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(1024)):
            with T.sblock("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv27[v_ax0, v_ax1, v_ax2])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                T_broadcast_to[v_ax0, v_ax1, v_ax2] = lv27[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def broadcast_to3(lv19: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(1024), T.int64(128)):
            with T.sblock("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv19[v_ax0, v_ax1, v_ax2])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                T_broadcast_to[v_ax0, v_ax1, v_ax2] = lv19[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def concatenate(A: T.Buffer((T.int64(32), T.int64(1008), T.int64(128)), "float32"), lv16: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), T_concat: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(1024), T.int64(128)):
            with T.sblock("T_concat"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv16[v_ax0, v_ax1 - T.int64(1008), v_ax2], A[v_ax0, v_ax1, v_ax2])
                T.writes(T_concat[v_ax0, v_ax1, v_ax2])
                T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(1008) <= v_ax1, lv16[v_ax0, v_ax1 - T.int64(1008), v_ax2], A[v_ax0, v_ax1, v_ax2])

    @T.prim_func(private=True)
    def divide(lv1: T.Buffer((T.int64(16),), "float32"), T_divide: T.Buffer((T.int64(16),), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0 in range(T.int64(16)):
            with T.sblock("T_divide"):
                v_ax0 = T.axis.spatial(T.int64(16), ax0)
                T.reads(lv1[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = lv1[v_ax0] / T.float32(4096.0)

    @T.prim_func(private=True)
    def divide1(x: T.Buffer((T.int64(16), T.int64(4096)), "float32"), lv4: T.Buffer((T.int64(16), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(16), T.int64(4096)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1 in T.grid(T.int64(16), T.int64(4096)):
            with T.sblock("T_divide"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1], lv4[v_ax0, T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1])
                T_divide[v_ax0, v_ax1] = x[v_ax0, v_ax1] / lv4[v_ax0, T.int64(0)]

    @T.prim_func(private=True)
    def divide2(lv24: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), lv26: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(1024)):
            with T.sblock("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv24[v_ax0, v_ax1, v_ax2], lv26[v_ax0, v_ax1, T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = lv24[v_ax0, v_ax1, v_ax2] / lv26[v_ax0, v_ax1, T.int64(0)]

    @T.prim_func(private=True)
    def expand_dims(lv3: T.Buffer((T.int64(16),), "float32"), expand_dims: T.Buffer((T.int64(16), T.int64(1)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1 in T.grid(T.int64(16), T.int64(1)):
            with T.sblock("expand_dims"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv3[v_i0])
                T.writes(expand_dims[v_i0, v_i1])
                expand_dims[v_i0, v_i1] = lv3[v_i0]

    @T.prim_func(private=True)
    def expand_dims1(lv25: T.Buffer((T.int64(32), T.int64(16)), "float32"), expand_dims: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2 in T.grid(T.int64(32), T.int64(16), T.int64(1)):
            with T.sblock("expand_dims"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(lv25[v_i0, v_i1])
                T.writes(expand_dims[v_i0, v_i1, v_i2])
                expand_dims[v_i0, v_i1, v_i2] = lv25[v_i0, v_i1]

    @T.prim_func(private=True)
    def matmul(lv5: T.Buffer((T.int64(16), T.int64(4096)), "float32"), lv6: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), matmul: T.Buffer((T.int64(16), T.int64(4096)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, k in T.grid(T.int64(16), T.int64(4096), T.int64(4096)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv5[v_i0, v_k], lv6[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + lv5[v_i0, v_k] * lv6[v_k, v_i1]

    @T.prim_func(private=True)
    def matmul1(lv21: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), lv22: T.Buffer((T.int64(32), T.int64(128), T.int64(1024)), "float32"), matmul: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2, k in T.grid(T.int64(32), T.int64(16), T.int64(1024), T.int64(128)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv21[v_i0, v_i1, v_k], lv22[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + lv21[v_i0, v_i1, v_k] * lv22[v_i0, v_k, v_i2]

    @T.prim_func(private=True)
    def matmul2(lv28: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), lv29: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2, k in T.grid(T.int64(32), T.int64(16), T.int64(128), T.int64(1024)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv28[v_i0, v_i1, v_k], lv29[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + lv28[v_i0, v_i1, v_k] * lv29[v_i0, v_k, v_i2]

    @T.prim_func(private=True)
    def multiply(x: T.Buffer((T.int64(16), T.int64(4096)), "float32"), x_1: T.Buffer((T.int64(16), T.int64(4096)), "float32"), T_multiply: T.Buffer((T.int64(16), T.int64(4096)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1 in T.grid(T.int64(16), T.int64(4096)):
            with T.sblock("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1], x_1[v_ax0, v_ax1])
                T.writes(T_multiply[v_ax0, v_ax1])
                T_multiply[v_ax0, v_ax1] = x[v_ax0, v_ax1] * x_1[v_ax0, v_ax1]

    @T.prim_func(private=True)
    def reshape(lv7: T.Buffer((T.int64(16), T.int64(4096)), "float32"), T_reshape: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(32), T.int64(128)):
            with T.sblock("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv7[((v_ax1 * T.int64(128) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(16), (v_ax1 * T.int64(128) + v_ax2) % T.int64(4096)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = lv7[((v_ax1 * T.int64(128) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(16), (v_ax1 * T.int64(128) + v_ax2) % T.int64(4096)]

    @T.prim_func(private=True)
    def reshape1(lv31: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32"), T_reshape: T.Buffer((T.int64(16), T.int64(4096)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1 in T.grid(T.int64(16), T.int64(4096)):
            with T.sblock("T_reshape"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv31[(v_ax1 // T.int64(4096) + v_ax0) % T.int64(16), v_ax1 % T.int64(4096) // T.int64(128), v_ax1 % T.int64(128)])
                T.writes(T_reshape[v_ax0, v_ax1])
                T_reshape[v_ax0, v_ax1] = lv31[(v_ax1 // T.int64(4096) + v_ax0) % T.int64(16), v_ax1 % T.int64(4096) // T.int64(128), v_ax1 % T.int64(128)]

    @T.prim_func(private=True)
    def sum(lv: T.Buffer((T.int64(16), T.int64(4096)), "float32"), lv_red: T.Buffer((T.int64(16),), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, k1 in T.grid(T.int64(16), T.int64(4096)):
            with T.sblock("lv_red"):
                v_ax0, v_k1 = T.axis.remap("SR", [ax0, k1])
                T.reads(lv[v_ax0, v_k1])
                T.writes(lv_red[v_ax0])
                with T.init():
                    lv_red[v_ax0] = T.float32(0.0)
                lv_red[v_ax0] = lv_red[v_ax0] + lv[v_ax0, v_k1]

    @T.prim_func(private=True)
    def sum1(lv24: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), lv24_red: T.Buffer((T.int64(32), T.int64(16)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, k2 in T.grid(T.int64(32), T.int64(16), T.int64(1024)):
            with T.sblock("lv24_red"):
                v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
                T.reads(lv24[v_ax0, v_ax1, v_k2])
                T.writes(lv24_red[v_ax0, v_ax1])
                with T.init():
                    lv24_red[v_ax0, v_ax1] = T.float32(0.0)
                lv24_red[v_ax0, v_ax1] = lv24_red[v_ax0, v_ax1] + lv24[v_ax0, v_ax1, v_k2]

    @T.prim_func(private=True)
    def tir_exp(lv23: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), compute: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2 in T.grid(T.int64(32), T.int64(16), T.int64(1024)):
            with T.sblock("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(lv23[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.exp(lv23[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def tir_sqrt(lv2: T.Buffer((T.int64(16),), "float32"), compute: T.Buffer((T.int64(16),), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0 in range(T.int64(16)):
            with T.sblock("compute"):
                v_i0 = T.axis.spatial(T.int64(16), i0)
                T.reads(lv2[v_i0])
                T.writes(compute[v_i0])
                compute[v_i0] = T.sqrt(lv2[v_i0])

    @T.prim_func(private=True)
    def transpose(p_q_proj_weight: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), T_transpose: T.Buffer((T.int64(4096), T.int64(4096)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
            with T.sblock("T_transpose"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(p_q_proj_weight[v_ax1, v_ax0])
                T.writes(T_transpose[v_ax0, v_ax1])
                T_transpose[v_ax0, v_ax1] = p_q_proj_weight[v_ax1, v_ax0]

    @T.prim_func(private=True)
    def transpose1(lv12: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(128)):
            with T.sblock("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv12[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = lv12[v_ax1, v_ax0, v_ax2]

    @T.prim_func(private=True)
    def transpose2(lv18: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(32), T.int64(128), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(128), T.int64(1024)):
            with T.sblock("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv18[v_ax0, v_ax2, v_ax1])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = lv18[v_ax0, v_ax2, v_ax1]

    @T.prim_func(private=True)
    def transpose3(lv30: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(32), T.int64(128)):
            with T.sblock("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv30[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = lv30[v_ax1, v_ax0, v_ax2]

    @R.function
    def main(x: R.Tensor((16, 4096), dtype="float32"), p_q_proj_weight: R.Tensor((4096, 4096), dtype="float32"), p_k_proj_weight: R.Tensor((4096, 4096), dtype="float32"), p_v_proj_weight: R.Tensor((4096, 4096), dtype="float32")) -> R.Tuple(R.Tensor((16, 4096), dtype="float32")):
        R.func_attr({"num_input": 1, "params": [metadata["ffi.Tensor"][0], metadata["ffi.Tensor"][1], metadata["ffi.Tensor"][2]]})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.multiply, (x, x), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            lv1 = R.call_tir(cls.sum, (lv,), out_sinfo=R.Tensor((16,), dtype="float32"))
            lv2 = R.call_tir(cls.divide, (lv1,), out_sinfo=R.Tensor((16,), dtype="float32"))
            lv3 = R.call_tir(cls.tir_sqrt, (lv2,), out_sinfo=R.Tensor((16,), dtype="float32"))
            lv4 = R.call_tir(cls.expand_dims, (lv3,), out_sinfo=R.Tensor((16, 1), dtype="float32"))
            lv5 = R.call_tir(cls.divide1, (x, lv4), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            lv6 = R.call_tir(cls.transpose, (p_q_proj_weight,), out_sinfo=R.Tensor((4096, 4096), dtype="float32"))
            lv7 = R.call_tir(cls.matmul, (lv5, lv6), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            lv8 = R.call_tir(cls.transpose, (p_k_proj_weight,), out_sinfo=R.Tensor((4096, 4096), dtype="float32"))
            lv9 = R.call_tir(cls.matmul, (lv5, lv8), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            lv10 = R.call_tir(cls.transpose, (p_v_proj_weight,), out_sinfo=R.Tensor((4096, 4096), dtype="float32"))
            lv11 = R.call_tir(cls.matmul, (lv5, lv10), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            lv12 = R.call_tir(cls.reshape, (lv7,), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
            lv13 = R.call_tir(cls.reshape, (lv9,), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
            lv14 = R.call_tir(cls.reshape, (lv11,), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
            lv15 = R.call_tir(cls.transpose1, (lv12,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv16 = R.call_tir(cls.transpose1, (lv13,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv17 = R.call_tir(cls.transpose1, (lv14,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv18 = R.call_tir(cls.concatenate, (metadata["relax.expr.Constant"][0], lv16), out_sinfo=R.Tensor((32, 1024, 128), dtype="float32"))
            lv19 = R.call_tir(cls.concatenate, (metadata["relax.expr.Constant"][1], lv17), out_sinfo=R.Tensor((32, 1024, 128), dtype="float32"))
            lv20 = R.call_tir(cls.transpose2, (lv18,), out_sinfo=R.Tensor((32, 128, 1024), dtype="float32"))
            lv21 = R.call_tir(cls.broadcast_to, (lv15,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv22 = R.call_tir(cls.broadcast_to1, (lv20,), out_sinfo=R.Tensor((32, 128, 1024), dtype="float32"))
            lv23 = R.call_tir(cls.matmul1, (lv21, lv22), out_sinfo=R.Tensor((32, 16, 1024), dtype="float32"))
            lv24 = R.call_tir(cls.tir_exp, (lv23,), out_sinfo=R.Tensor((32, 16, 1024), dtype="float32"))
            lv25 = R.call_tir(cls.sum1, (lv24,), out_sinfo=R.Tensor((32, 16), dtype="float32"))
            lv26 = R.call_tir(cls.expand_dims1, (lv25,), out_sinfo=R.Tensor((32, 16, 1), dtype="float32"))
            lv27 = R.call_tir(cls.divide2, (lv24, lv26), out_sinfo=R.Tensor((32, 16, 1024), dtype="float32"))
            lv28 = R.call_tir(cls.broadcast_to2, (lv27,), out_sinfo=R.Tensor((32, 16, 1024), dtype="float32"))
            lv29 = R.call_tir(cls.broadcast_to3, (lv19,), out_sinfo=R.Tensor((32, 1024, 128), dtype="float32"))
            lv30 = R.call_tir(cls.matmul2, (lv28, lv29), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv31 = R.call_tir(cls.transpose3, (lv30,), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
            lv32 = R.call_tir(cls.reshape1, (lv31,), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            gv: R.Tuple(R.Tensor((16, 4096), dtype="float32")) = (lv32,)
            R.output(gv)
        return gv

# Metadata omitted. Use show_meta=True in script() method to show it.