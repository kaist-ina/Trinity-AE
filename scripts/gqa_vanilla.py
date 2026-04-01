import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

import torch
import torch.nn as nn


class GQAVanilla(nn.Module):
    def __init__(
        self,
        M,
        QH,
        D,
        P,
        cache_K,
        cache_V,
        W_q=None,
        W_k=None,
        W_v=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.M = M
        self.QH = QH
        self.D = D
        self.P = P
        self.N = QH * D
        self.device = device
        self.dtype = dtype

        self.q_proj = nn.Linear(self.N, self.N, bias=False)
        self.k_proj = nn.Linear(self.N, self.N, bias=False)
        self.v_proj = nn.Linear(self.N, self.N, bias=False)

        if W_q is not None:
            self.q_proj.weight.data = W_q.T.to(device=device, dtype=dtype)
        if W_k is not None:
            self.k_proj.weight.data = W_k.T.to(device=device, dtype=dtype)
        if W_v is not None:
            self.v_proj.weight.data = W_v.T.to(device=device, dtype=dtype)

        self.register_buffer("cache_K", cache_K.to(device))
        self.register_buffer("cache_V", cache_V.to(device))

    def forward(self, X):
        q1 = self.q_proj(X)
        k1 = self.k_proj(X)
        v1 = self.v_proj(X)

        q2 = q1.view(self.M, self.QH, self.D)
        k2 = k1.view(self.M, self.QH, self.D)
        v2 = v1.view(self.M, self.QH, self.D)

        q = q2.permute(1, 0, 2)
        k = k2.permute(1, 0, 2)
        v = v2.permute(1, 0, 2)

        self.cache_K[:, self.P : self.P + self.M, :] = k
        self.cache_V[:, self.P : self.P + self.M, :] = v

        c = torch.matmul(q, self.cache_K.permute(0, 2, 1))
        c_exp = torch.exp(c)
        c_sum = c_exp.sum(dim=2)
        c_div = c_exp / c_sum.unsqueeze(-1)

        o = torch.matmul(c_div, self.cache_V)
        o1 = o.permute(1, 0, 2)
        o2 = o1.contiguous().view(self.M, self.N)
        return o2


if __name__ == "__main__":
    import trinity

    M, QH, D, P = 16, 32, 128, 1008
    N = QH * D

    X = torch.randn((M, N))
    K_cache = torch.randn((QH, P + M, D))
    V_cache = torch.randn((QH, P + M, D))

    model = GQAVanilla(M, QH, D, P, K_cache, V_cache)
    result = trinity.optimize(model, X, basename="gqa_vanilla", verbose=True)
