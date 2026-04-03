import torch
import torch.nn as nn
import ir.AST as T
from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
from core.from_tir import build_primfunc_nodes
from core.from_rir import build_main_func
from core.sequantialize import sequentialize_main_func
from core.to_ir import calls_to_ir, filter_identity_and_apply_alias
from utils.io_utils import format_primfunc_nodes, format_main_func, export_main_func
from utils.ir_utils import inline_elementwise_op_calls, inline_shape_op_calls, bind_main_func_calls, normalize_main_func_axes
from utils.test_utils import validate_main_func_errors
from utils.pipeline import export_model_ir

class FFN(nn.Module):
    def __init__(self, M, N, N4, WO=None, WFF1a=None, WFF1b=None, WFF2=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.N4 = N4
        self.device = device
        self.dtype = dtype

        # nn.Linear layers (bias=False)
        self.fc_o = nn.Linear(N, N, bias=False)
        self.fc_ff1a = nn.Linear(N, N4, bias=False)
        self.fc_ff1b = nn.Linear(N, N4, bias=False)
        self.fc_ff2 = nn.Linear(N4, N, bias=False)

        # Initialize weights with provided values
        # nn.Linear weight shape is (out_features, in_features)
        # torch.matmul(x, W) = x @ W, but nn.Linear does x @ W.T
        # So we need to transpose the weights
        if WO is not None:
            self.fc_o.weight.data = WO.T.to(device=device, dtype=dtype)
        if WFF1a is not None:
            self.fc_ff1a.weight.data = WFF1a.T.to(device=device, dtype=dtype)
        if WFF1b is not None:
            self.fc_ff1b.weight.data = WFF1b.T.to(device=device, dtype=dtype)
        if WFF2 is not None:
            self.fc_ff2.weight.data = WFF2.T.to(device=device, dtype=dtype)

    def forward(self, O2, X):
        attn_O1 = self.fc_o(O2)
        attn_O2 = attn_O1 + X
        attn_O3 = attn_O2.pow(2).mean(-1, keepdim=True)
        attn_O_norm = attn_O2 * torch.rsqrt(attn_O3)

        FF1a = self.fc_ff1a(attn_O_norm)
        FF1b = self.fc_ff1b(attn_O_norm)
        FF1b_silu = FF1b * torch.sigmoid(FF1b)

        FF1 = FF1a * FF1b_silu
        FF2 = self.fc_ff2(FF1)

        return FF2
    
def to_relax(model, example_input1, example_input2):
    """PyTorch 모델을 Relax IR로 변환"""
    model.eval()
    with torch.no_grad():
        # export 함수는 tuple of tensors를 받아야 함
        # if not isinstance(example_input, tuple):
        example_input = (example_input1, example_input2,)
        exported = export(model, example_input)
    return from_exported_program(exported, keep_params_as_input=True)


def to_tir(relax_mod):
    """Relax IR을 TIR로 lowering"""
    return relax.transform.LegalizeOps()(relax_mod)

def build_model_and_inputs():
    import os

    M = 16
    D = 128
    N = 4096
    N4 = 16384
    P = 1024
    H = 32

    device = torch.device('cpu')  # TVM export는 CPU에서 더 안정적
    dtype = torch.float32  # float16은 export 시 문제가 있을 수 있음

    FF1 = torch.zeros(M, N4, dtype=dtype, device=device)
    FF2 = torch.zeros(M, N4, dtype=dtype, device=device)
    WFF1a = torch.randn(N, N4, dtype=dtype, device=device)
    WFF1b = torch.randn(N, N4, dtype=dtype, device=device)
    WFF2 = torch.randn(N4, N, dtype=dtype, device=device)
    WO = torch.randn(N, N, dtype=dtype, device=device)
    attn_O2 = torch.zeros(M, N, dtype=dtype, device=device)

    X = torch.randn((M, N), device=device, dtype=dtype)
    O2 = torch.randn(M, N, dtype=dtype, device=device)

    model = FFN(M, N, N4, WO=WO, WFF1a=WFF1a, WFF1b=WFF1b, WFF2=WFF2, device=device, dtype=dtype)

    # forward(O2, X) 순서에 맞춰 전달
    example_inputs = (O2, X,)
    return {
        "model": model,
        "example_inputs": example_inputs,
        "inline_shape_op": True,
        "inline_elementwise_op": True,
        "remove_short_loop_threshold": 16,
        "decompose_nested_op_ratio": 0.0,
    }
if __name__ == "__main__":
    cfg = build_model_and_inputs()
    export_model_ir(
        cfg["model"],
        cfg["example_inputs"],
        inline_shape_op=cfg.get("inline_shape_op", True),
        inline_elementwise_op=cfg.get("inline_elementwise_op", True),
        remove_short_loop_threshold=cfg.get("remove_short_loop_threshold", 64),
        decompose_nested_op_ratio=cfg.get("decompose_nested_op_ratio", 0.0),
    )
