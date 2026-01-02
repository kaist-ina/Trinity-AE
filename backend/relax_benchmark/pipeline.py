from typing import List

import tvm
from tvm import relax, dlight


@tvm.transform.module_pass(opt_level=0)
def opt_gpu(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
    """
    GPU optimization pipeline for Relax IR.

    Phases:
        1. Pattern fusion & Library integration (cuBLAS, cuTLASS)
        2. Lowering to TIR
        3. Operator fusion
        4. GPU scheduling via dlight
        5. Lowering to VM bytecode
    """

    try:
        import tvm.relax.backend.cuda.cublas as _cublas
        mod = _cublas.partition_for_cublas(mod)
        print("[Pipeline] cuBLAS enabled")
    except Exception:
        print("[Pipeline] cuBLAS not available")
    
    mod = relax.transform.RunCodegen()(mod)

    # Phase 1-5. TIR lowering, fusion, scheduling, VM lowering
    seq = tvm.transform.Sequential([
        # Phase 1. Pattern fusion & External library integration
        relax.transform.FuseTransposeMatmul(),
        # Phase 2. Lowering to TIR
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),

        # Phase 3. Operator fusion
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
        relax.transform.DeadCodeElimination(),

        # Phase 4. GPU scheduling
        dlight.ApplyDefaultSchedule(
            dlight.gpu.Matmul(),
            dlight.gpu.GEMV(),
            dlight.gpu.Reduction(),
            dlight.gpu.GeneralReduction(),
            dlight.gpu.Fallback(),
        ),

        # Phase 5. Lowering to VM bytecode
        relax.transform.RewriteDataflowReshape(),
        relax.transform.ToNonDataflow(),
        relax.transform.RemovePurityChecking(),
        relax.transform.CallTIRRewrite(),
        relax.transform.StaticPlanBlockMemory(),
        relax.transform.RewriteCUDAGraph(),
        relax.transform.LowerAllocTensor(),
        relax.transform.KillAfterLastUse(),
        relax.transform.LowerRuntimeBuiltin(),
        relax.transform.VMShapeLower(),
        relax.transform.AttachGlobalSymbol(),
    ])

    mod = seq(mod)
    return mod
