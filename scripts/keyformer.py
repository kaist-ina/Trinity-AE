import sys
from pathlib import Path

# Setup paths for trinity and frontend imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

import torch
import torch.nn as nn


class KeyformerAttn(nn.Module):
    def __init__(self, M, H, D, P, cache_K, cache_V, tau, noise, device=None, dtype=None):
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
        self.register_buffer("tau", tau.to(device))
        self.register_buffer("noise", noise.to(device))

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

        k_cache = torch.cat([self.cache_K, k], dim=1)
        v_cache = torch.cat([self.cache_V, v], dim=1)

        c = torch.matmul(q, k_cache.permute(0, 2, 1))
        c_perturb = (c + self.noise) / self.tau

        c_exp = torch.exp(c)
        c_exp_perturb = torch.exp(c_perturb)

        c_sum = c_exp.sum(dim=2)
        c_sum_perturb = c_exp_perturb.sum(dim=2)

        c_div = c_exp / c_sum.unsqueeze(-1)
        c_div_perturb = c_exp_perturb / c_sum_perturb.unsqueeze(-1)
        c_out = c_div_perturb.sum(dim=1)

        o = torch.matmul(c_div, v_cache)
        o1 = o.permute(1, 0, 2)
        o2 = o1.contiguous().view(self.M, self.N)
        return o2, c_out


if __name__ == "__main__":
    import trinity

    M, H, D, P = 16, 32, 128, 1008
    N = H * D

    X = torch.randn((M, N))
    K_cache = torch.randn((H, P, D))
    V_cache = torch.randn((H, P, D))
    tau = torch.tensor(1.5)
    noise = torch.randn((H, M, P + M))

    model = KeyformerAttn(M, H, D, P, K_cache, V_cache, tau, noise)
    result = trinity.optimize(model, X, basename="keyformer", verbose=True)
