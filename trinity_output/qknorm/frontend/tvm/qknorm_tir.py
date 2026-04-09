# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def broadcast_to(lv18: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(128)):
            with T.sblock("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv18[v_ax0, v_ax1, v_ax2])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                T_broadcast_to[v_ax0, v_ax1, v_ax2] = lv18[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def broadcast_to1(lv32: T.Buffer((T.int64(32), T.int64(128), T.int64(1024)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(128), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(128), T.int64(1024)):
            with T.sblock("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv32[v_ax0, v_ax1, v_ax2])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                T_broadcast_to[v_ax0, v_ax1, v_ax2] = lv32[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def broadcast_to2(lv37: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(1024)):
            with T.sblock("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv37[v_ax0, v_ax1, v_ax2])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                T_broadcast_to[v_ax0, v_ax1, v_ax2] = lv37[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def broadcast_to3(lv30: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), T_broadcast_to: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(1024), T.int64(128)):
            with T.sblock("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv30[v_ax0, v_ax1, v_ax2])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                T_broadcast_to[v_ax0, v_ax1, v_ax2] = lv30[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def divide(lv10: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(1)):
            with T.sblock("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv10[v_ax0, v_ax1, v_ax2])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = lv10[v_ax0, v_ax1, v_ax2] / T.float32(128.0)

    @T.prim_func(private=True)
    def divide1(lv6: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), lv12: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(128)):
            with T.sblock("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv6[v_ax0, v_ax1, v_ax2], lv12[v_ax0, v_ax1, T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = lv6[v_ax0, v_ax1, v_ax2] / lv12[v_ax0, v_ax1, T.int64(0)]

    @T.prim_func(private=True)
    def divide2(lv35: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), lv36: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32"), T_divide: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(1024)):
            with T.sblock("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv35[v_ax0, v_ax1, v_ax2], lv36[v_ax0, v_ax1, T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = lv35[v_ax0, v_ax1, v_ax2] / lv36[v_ax0, v_ax1, T.int64(0)]

    @T.prim_func(private=True)
    def matmul(x: T.Buffer((T.int64(16), T.int64(4096)), "float32"), B: T.Buffer((T.int64(4096), T.int64(4096)), "float32"), matmul: T.Buffer((T.int64(16), T.int64(4096)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, k in T.grid(T.int64(16), T.int64(4096), T.int64(4096)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], B[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0.0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + x[v_i0, v_k] * B[v_k, v_i1]

    @T.prim_func(private=True)
    def matmul1(lv31: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), lv33: T.Buffer((T.int64(32), T.int64(128), T.int64(1024)), "float32"), matmul: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2, k in T.grid(T.int64(32), T.int64(16), T.int64(1024), T.int64(128)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv31[v_i0, v_i1, v_k], lv33[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + lv31[v_i0, v_i1, v_k] * lv33[v_i0, v_k, v_i2]

    @T.prim_func(private=True)
    def matmul2(lv38: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), lv39: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2, k in T.grid(T.int64(32), T.int64(16), T.int64(128), T.int64(1024)):
            with T.sblock("matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv38[v_i0, v_i1, v_k], lv39[v_i0, v_k, v_i2])
                T.writes(matmul[v_i0, v_i1, v_i2])
                with T.init():
                    matmul[v_i0, v_i1, v_i2] = T.float32(0.0)
                matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + lv38[v_i0, v_i1, v_k] * lv39[v_i0, v_k, v_i2]

    @T.prim_func(private=True)
    def multiply(lv6: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), lv6_1: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), T_multiply: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(128)):
            with T.sblock("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv6[v_ax0, v_ax1, v_ax2], lv6_1[v_ax0, v_ax1, v_ax2])
                T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                T_multiply[v_ax0, v_ax1, v_ax2] = lv6[v_ax0, v_ax1, v_ax2] * lv6_1[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def reshape(lv: T.Buffer((T.int64(16), T.int64(4096)), "float32"), T_reshape: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(32), T.int64(128)):
            with T.sblock("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv[((v_ax1 * T.int64(128) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(16), (v_ax1 * T.int64(128) + v_ax2) % T.int64(4096)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = lv[((v_ax1 * T.int64(128) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(16), (v_ax1 * T.int64(128) + v_ax2) % T.int64(4096)]

    @T.prim_func(private=True)
    def reshape1(lv41: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32"), T_reshape: T.Buffer((T.int64(16), T.int64(4096)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1 in T.grid(T.int64(16), T.int64(4096)):
            with T.sblock("T_reshape"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv41[(v_ax1 // T.int64(4096) + v_ax0) % T.int64(16), v_ax1 % T.int64(4096) // T.int64(128), v_ax1 % T.int64(128)])
                T.writes(T_reshape[v_ax0, v_ax1])
                T_reshape[v_ax0, v_ax1] = lv41[(v_ax1 // T.int64(4096) + v_ax0) % T.int64(16), v_ax1 % T.int64(4096) // T.int64(128), v_ax1 % T.int64(128)]

    @T.prim_func(private=True)
    def slice_scatter(lv21: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), lv20: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), compute: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2 in T.grid(T.int64(32), T.int64(16), T.int64(128)):
            with T.sblock("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(lv20[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = lv20[v_i0, v_i1, v_i2]

    @T.prim_func(private=True)
    def slice_scatter1(A: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), lv22: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), T_where: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        T_full = T.sblock_alloc_buffer((T.int64(1024),), "bool")
        T_arange = T.sblock_alloc_buffer((T.int64(1024),), "int64")
        T_greater_equal = T.sblock_alloc_buffer((T.int64(1024),), "bool")
        T_logical_and = T.sblock_alloc_buffer((T.int64(1024),), "bool")
        T_reshape = T.sblock_alloc_buffer((T.int64(1), T.int64(1024), T.int64(1)), "bool")
        T_broadcast_to = T.sblock_alloc_buffer((T.int64(32), T.int64(1024), T.int64(128)), "bool")
        T_subtract = T.sblock_alloc_buffer((T.int64(1024),), "int64")
        T_add = T.sblock_alloc_buffer((T.int64(1024),), "int64")
        T_floor_divide = T.sblock_alloc_buffer((T.int64(1024),), "int64")
        compute = T.sblock_alloc_buffer((T.int64(1024),), "int64")
        T_take = T.sblock_alloc_buffer((T.int64(32), T.int64(1024), T.int64(128)))
        for ax0 in range(T.int64(1024)):
            with T.sblock("T_full"):
                v_ax0 = T.axis.spatial(T.int64(1024), ax0)
                T.reads()
                T.writes(T_full[v_ax0])
                T_full[v_ax0] = T.bool(True)
        for ax0 in range(T.int64(1024)):
            with T.sblock("T_arange"):
                v_ax0 = T.axis.spatial(T.int64(1024), ax0)
                T.reads()
                T.writes(T_arange[v_ax0])
                T_arange[v_ax0] = T.Cast("int64", v_ax0)
        for ax0 in range(T.int64(1024)):
            with T.sblock("T_greater_equal"):
                v_ax0 = T.axis.spatial(T.int64(1024), ax0)
                T.reads(T_arange[v_ax0])
                T.writes(T_greater_equal[v_ax0])
                T_greater_equal[v_ax0] = T.int64(1008) <= T_arange[v_ax0]
        for ax0 in range(T.int64(1024)):
            with T.sblock("T_logical_and"):
                v_ax0 = T.axis.spatial(T.int64(1024), ax0)
                T.reads(T_full[v_ax0], T_greater_equal[v_ax0])
                T.writes(T_logical_and[v_ax0])
                T_logical_and[v_ax0] = T_full[v_ax0] and T_greater_equal[v_ax0]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1024), T.int64(1)):
            with T.sblock("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_logical_and[(v_ax1 + v_ax2) % T.int64(1024)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = T_logical_and[(v_ax1 + v_ax2) % T.int64(1024)]
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(1024), T.int64(128)):
            with T.sblock("T_broadcast_to"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_reshape[T.int64(0), v_ax1, T.int64(0)])
                T.writes(T_broadcast_to[v_ax0, v_ax1, v_ax2])
                T_broadcast_to[v_ax0, v_ax1, v_ax2] = T_reshape[T.int64(0), v_ax1, T.int64(0)]
        for ax0 in range(T.int64(1024)):
            with T.sblock("T_subtract"):
                v_ax0 = T.axis.spatial(T.int64(1024), ax0)
                T.reads(T_arange[v_ax0])
                T.writes(T_subtract[v_ax0])
                T_subtract[v_ax0] = T_arange[v_ax0] - T.int64(1008)
        for ax0 in range(T.int64(1024)):
            with T.sblock("T_add"):
                v_ax0 = T.axis.spatial(T.int64(1024), ax0)
                T.reads(T_subtract[v_ax0])
                T.writes(T_add[v_ax0])
                T_add[v_ax0] = T_subtract[v_ax0]
        for ax0 in range(T.int64(1024)):
            with T.sblock("T_floor_divide"):
                v_ax0 = T.axis.spatial(T.int64(1024), ax0)
                T.reads(T_add[v_ax0])
                T.writes(T_floor_divide[v_ax0])
                T_floor_divide[v_ax0] = T_add[v_ax0]
        for i0 in range(T.int64(1024)):
            with T.sblock("compute"):
                v_i0 = T.axis.spatial(T.int64(1024), i0)
                T.reads(T_floor_divide[v_i0])
                T.writes(compute[v_i0])
                compute[v_i0] = T.max(T.min(T_floor_divide[v_i0], T.int64(1023)), T.int64(0))
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(1024), T.int64(128)):
            with T.sblock("T_take"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv22[v_ax0, compute[v_ax1], v_ax2], compute[v_ax1])
                T.writes(T_take[v_ax0, v_ax1, v_ax2])
                T_take[v_ax0, v_ax1, v_ax2] = lv22[v_ax0, compute[v_ax1], v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(1024), T.int64(128)):
            with T.sblock("T_where"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_broadcast_to[v_ax0, v_ax1, v_ax2], T_take[v_ax0, v_ax1, v_ax2], A[v_ax0, v_ax1, v_ax2])
                T.writes(T_where[v_ax0, v_ax1, v_ax2])
                T_where[v_ax0, v_ax1, v_ax2] = T.Select(T.int64(0) < T.Cast("int64", T_broadcast_to[v_ax0, v_ax1, v_ax2]), T_take[v_ax0, v_ax1, v_ax2], A[v_ax0, v_ax1, v_ax2])

    @T.prim_func(private=True)
    def slice_scatter2(A: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), lv23: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), compute: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2 in T.grid(T.int64(32), T.int64(1024), T.int64(128)):
            with T.sblock("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(lv23[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = lv23[v_i0, v_i1, v_i2]

    @T.prim_func(private=True)
    def strided_slice(A: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(128)):
            with T.sblock("T_strided_slice_with_axes"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1 + T.int64(1008), v_ax2])
                T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2])
                T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1 + T.int64(1008), v_ax2]

    @T.prim_func(private=True)
    def sum(lv9: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), lv9_red: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2, k2 in T.grid(T.int64(32), T.int64(16), T.int64(1), T.int64(128)):
            with T.sblock("lv9_red"):
                v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                T.reads(lv9[v_ax0, v_ax1, v_k2])
                T.writes(lv9_red[v_ax0, v_ax1, v_ax2])
                with T.init():
                    lv9_red[v_ax0, v_ax1, v_ax2] = T.float32(0.0)
                lv9_red[v_ax0, v_ax1, v_ax2] = lv9_red[v_ax0, v_ax1, v_ax2] + lv9[v_ax0, v_ax1, v_k2]

    @T.prim_func(private=True)
    def sum1(lv35: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), lv35_red: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2, k2 in T.grid(T.int64(32), T.int64(16), T.int64(1), T.int64(1024)):
            with T.sblock("lv35_red"):
                v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                T.reads(lv35[v_ax0, v_ax1, v_k2])
                T.writes(lv35_red[v_ax0, v_ax1, v_ax2])
                with T.init():
                    lv35_red[v_ax0, v_ax1, v_ax2] = T.float32(0.0)
                lv35_red[v_ax0, v_ax1, v_ax2] = lv35_red[v_ax0, v_ax1, v_ax2] + lv35[v_ax0, v_ax1, v_k2]

    @T.prim_func(private=True)
    def tir_exp(lv34: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32"), compute: T.Buffer((T.int64(32), T.int64(16), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2 in T.grid(T.int64(32), T.int64(16), T.int64(1024)):
            with T.sblock("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(lv34[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.exp(lv34[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def tir_sqrt(lv11: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32"), compute: T.Buffer((T.int64(32), T.int64(16), T.int64(1)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i0, i1, i2 in T.grid(T.int64(32), T.int64(16), T.int64(1)):
            with T.sblock("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(lv11[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.sqrt(lv11[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def transpose(lv3: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(16), T.int64(128)):
            with T.sblock("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv3[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = lv3[v_ax1, v_ax0, v_ax2]

    @T.prim_func(private=True)
    def transpose1(lv24: T.Buffer((T.int64(32), T.int64(1024), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(32), T.int64(128), T.int64(1024)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(32), T.int64(128), T.int64(1024)):
            with T.sblock("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv24[v_ax0, v_ax2, v_ax1])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = lv24[v_ax0, v_ax2, v_ax1]

    @T.prim_func(private=True)
    def transpose2(lv40: T.Buffer((T.int64(32), T.int64(16), T.int64(128)), "float32"), T_transpose: T.Buffer((T.int64(16), T.int64(32), T.int64(128)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(32), T.int64(128)):
            with T.sblock("T_transpose"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv40[v_ax1, v_ax0, v_ax2])
                T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
                T_transpose[v_ax0, v_ax1, v_ax2] = lv40[v_ax1, v_ax0, v_ax2]

    @R.function
    def main(x: R.Tensor((16, 4096), dtype="float32")) -> R.Tuple(R.Tensor((32, 1024, 128), dtype="float32"), R.Tensor((32, 1024, 128), dtype="float32"), R.Tensor((16, 4096), dtype="float32")):
        R.func_attr({"num_input": 1, "params": []})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.matmul, (x, metadata["relax.expr.Constant"][0]), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            lv1 = R.call_tir(cls.matmul, (x, metadata["relax.expr.Constant"][1]), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            lv2 = R.call_tir(cls.matmul, (x, metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            lv3 = R.call_tir(cls.reshape, (lv,), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
            lv4 = R.call_tir(cls.reshape, (lv1,), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
            lv5 = R.call_tir(cls.reshape, (lv2,), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
            lv6 = R.call_tir(cls.transpose, (lv3,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv7 = R.call_tir(cls.transpose, (lv4,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv8 = R.call_tir(cls.transpose, (lv5,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv9 = R.call_tir(cls.multiply, (lv6, lv6), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv10 = R.call_tir(cls.sum, (lv9,), out_sinfo=R.Tensor((32, 16, 1), dtype="float32"))
            lv11 = R.call_tir(cls.divide, (lv10,), out_sinfo=R.Tensor((32, 16, 1), dtype="float32"))
            lv12 = R.call_tir(cls.tir_sqrt, (lv11,), out_sinfo=R.Tensor((32, 16, 1), dtype="float32"))
            lv13 = R.call_tir(cls.divide1, (lv6, lv12), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv14 = R.call_tir(cls.multiply, (lv7, lv7), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv15 = R.call_tir(cls.sum, (lv14,), out_sinfo=R.Tensor((32, 16, 1), dtype="float32"))
            lv16 = R.call_tir(cls.divide, (lv15,), out_sinfo=R.Tensor((32, 16, 1), dtype="float32"))
            lv17 = R.call_tir(cls.tir_sqrt, (lv16,), out_sinfo=R.Tensor((32, 16, 1), dtype="float32"))
            lv18 = R.call_tir(cls.divide1, (lv7, lv17), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv19 = R.call_tir(cls.strided_slice, (metadata["relax.expr.Constant"][3],), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv20 = R.call_tir(cls.broadcast_to, (lv18,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv21 = R.call_tir(cls.strided_slice, (metadata["relax.expr.Constant"][3],), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv22 = R.call_tir(cls.slice_scatter, (lv21, lv20), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv23 = R.call_tir(cls.slice_scatter1, (metadata["relax.expr.Constant"][3], lv22), out_sinfo=R.Tensor((32, 1024, 128), dtype="float32"))
            lv24 = R.call_tir(cls.slice_scatter2, (metadata["relax.expr.Constant"][3], lv23), out_sinfo=R.Tensor((32, 1024, 128), dtype="float32"))
            lv25 = R.call_tir(cls.strided_slice, (metadata["relax.expr.Constant"][4],), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv26 = R.call_tir(cls.broadcast_to, (lv8,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv27 = R.call_tir(cls.strided_slice, (metadata["relax.expr.Constant"][4],), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv28 = R.call_tir(cls.slice_scatter, (lv27, lv26), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv29 = R.call_tir(cls.slice_scatter1, (metadata["relax.expr.Constant"][4], lv28), out_sinfo=R.Tensor((32, 1024, 128), dtype="float32"))
            lv30 = R.call_tir(cls.slice_scatter2, (metadata["relax.expr.Constant"][4], lv29), out_sinfo=R.Tensor((32, 1024, 128), dtype="float32"))
            lv31 = R.call_tir(cls.broadcast_to, (lv13,), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv32 = R.call_tir(cls.transpose1, (lv24,), out_sinfo=R.Tensor((32, 128, 1024), dtype="float32"))
            lv33 = R.call_tir(cls.broadcast_to1, (lv32,), out_sinfo=R.Tensor((32, 128, 1024), dtype="float32"))
            lv34 = R.call_tir(cls.matmul1, (lv31, lv33), out_sinfo=R.Tensor((32, 16, 1024), dtype="float32"))
            lv35 = R.call_tir(cls.tir_exp, (lv34,), out_sinfo=R.Tensor((32, 16, 1024), dtype="float32"))
            lv36 = R.call_tir(cls.sum1, (lv35,), out_sinfo=R.Tensor((32, 16, 1), dtype="float32"))
            lv37 = R.call_tir(cls.divide2, (lv35, lv36), out_sinfo=R.Tensor((32, 16, 1024), dtype="float32"))
            lv38 = R.call_tir(cls.broadcast_to2, (lv37,), out_sinfo=R.Tensor((32, 16, 1024), dtype="float32"))
            lv39 = R.call_tir(cls.broadcast_to3, (lv30,), out_sinfo=R.Tensor((32, 1024, 128), dtype="float32"))
            lv40 = R.call_tir(cls.matmul2, (lv38, lv39), out_sinfo=R.Tensor((32, 16, 128), dtype="float32"))
            lv41 = R.call_tir(cls.transpose2, (lv40,), out_sinfo=R.Tensor((16, 32, 128), dtype="float32"))
            lv42 = R.call_tir(cls.reshape1, (lv41,), out_sinfo=R.Tensor((16, 4096), dtype="float32"))
            gv: R.Tuple(R.Tensor((32, 1024, 128), dtype="float32"), R.Tensor((32, 1024, 128), dtype="float32"), R.Tensor((16, 4096), dtype="float32")) = lv24, lv30, lv42
            R.output(gv)
        return gv

# Metadata omitted. Use show_meta=True in script() method to show it.