import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

import torch
import torch.nn as nn


class PreNormAttn(nn.Module):
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
        x2 = (X * X).sum(dim=1)
        x_norm = X / torch.sqrt(x2 / self.N).unsqueeze(1)

        q1 = self.q_proj(x_norm)
        k1 = self.k_proj(x_norm)
        v1 = self.v_proj(x_norm)

        q2 = q1.view(self.M, self.H, self.D)
        k2 = k1.view(self.M, self.H, self.D)
        v2 = v1.view(self.M, self.H, self.D)

        q = q2.permute(1, 0, 2)
        k = k2.permute(1, 0, 2)
        v = v2.permute(1, 0, 2)

        k_cache = torch.cat([self.cache_K, k], dim=1)
        v_cache = torch.cat([self.cache_V, v], dim=1)

        c = torch.matmul(q, k_cache.permute(0, 2, 1))
        c_exp = torch.exp(c)
        c_sum = c_exp.sum(dim=2)
        c_div = c_exp / c_sum.unsqueeze(-1)
        o = torch.matmul(c_div, v_cache)

        o1 = o.permute(1, 0, 2)
        o2 = o1.contiguous().view(self.M, self.N)
        return o2


if __name__ == "__main__":
    import trinity

    M, H, D, P = 16, 32, 128, 1008
    N = H * D

    X = torch.randn((M, N))
    K_cache = torch.randn((H, P, D))
    V_cache = torch.randn((H, P, D))

    model = PreNormAttn(M, H, D, P, K_cache, V_cache)
    result = trinity.optimize(model, X, basename="prenorm", verbose=True)
