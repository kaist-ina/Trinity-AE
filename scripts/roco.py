import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))



class RocoAttnWithStats(nn.Module):
    def __init__(self, M, H, D, P, cache_K, cache_V, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.H = H
        self.D = D
        self.P = P
        self.N = H * D
        self.device = device
        self.dtype = dtype

        self.q_proj = nn.Linear(self.N, self.N, bias=False)
        self.k_proj = nn.Linear(self.N, self.N, bias=False)
        self.v_proj = nn.Linear(self.N, self.N, bias=False)

        self.register_buffer("cache_K", cache_K.to(device))
        self.register_buffer("cache_V", cache_V.to(device))

    def forward(self, X):
        q1 = self.q_proj(X)
        k1 = self.k_proj(X)
        v1 = self.v_proj(X)

        q2 = q1.view(self.M, self.H, self.D)
        k2 = k1.view(self.M, self.H, self.D)
        v2 = v1.view(self.M, self.H, self.D)

        q = q2.permute(1, 0, 2)
        k = k2.permute(1, 0, 2)
        v = v2.permute(1, 0, 2)

        end = self.cache_K.size(1)
        start = end - k.size(1)
        self.cache_K[:, start:end, :] = k
        self.cache_V[:, start:end, :] = v

        c = torch.matmul(q, self.cache_K.permute(0, 2, 1))
        c_exp = torch.exp(c)
        c_sum = c_exp.sum(dim=2)
        c_div = c_exp / c_sum.unsqueeze(-1)

        o = torch.matmul(c_div, self.cache_V)
        c_out1 = c_div.sum(dim=1)
        c_out2 = (c_div * c_div).sum(dim=1)

        o1 = o.permute(1, 0, 2)
        o2 = o1.contiguous().view(self.M, self.N)
        return o2, c_out1, c_out2


def build_model_and_inputs():
    device = torch.device("cpu")
    dtype = torch.float32

    M = 16
    H = 32
    D = 128
    P = 1040

    N = H * D
    X = torch.randn((M, N), device=device, dtype=dtype)
    K_cache = torch.randn((H, P, D), device=device, dtype=dtype)
    V_cache = torch.randn((H, P, D), device=device, dtype=dtype)

    model = RocoAttnWithStats(M, H, D, P, K_cache, V_cache, device=device, dtype=dtype)

    print("Model created. Converting to Relax IR...")
    example_inputs = X
    return {
        "model": model,
        "example_inputs": example_inputs,
        "inline_shape_op": True,
        "inline_elementwise_op": True,
        "remove_short_loop_threshold": 24,
        "decompose_nested_op_ratio": 0.0,
    }


if __name__ == "__main__":
    import trinity
    cfg = build_model_and_inputs()
    result = trinity.optimize(cfg["model"], cfg["example_inputs"], basename="roco")
