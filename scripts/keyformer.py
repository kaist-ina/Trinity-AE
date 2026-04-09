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

        # self.q_proj = nn.Linear(self.N, self.N, bias=False)
        # self.k_proj = nn.Linear(self.N, self.N, bias=False)
        # self.v_proj = nn.Linear(self.N, self.N, bias=False)

        self.q_proj = torch.randn(N, N, device=device, dtype=dtype)
        self.k_proj = torch.randn(N, N, device=device, dtype=dtype)
        self.v_proj = torch.randn(N, N, device=device, dtype=dtype)

        self.register_buffer("cache_K", cache_K.to(device))
        self.register_buffer("cache_V", cache_V.to(device))
        self.register_buffer("tau", tau.to(device))
        self.register_buffer("noise", noise.to(device))

    def forward(self, X):
        q = torch.matmul(X, self.q_proj)
        k = torch.matmul(X, self.k_proj)
        v = torch.matmul(X, self.v_proj)

    
        # Reshape to multi-head
        q = q.view(self.M, self.H, self.D)  # (M, H, D)
        k = k.view(self.M, self.H, self.D)  # (M, H, D)
        v = v.view(self.M, self.H, self.D)  # (M, H, D)

        # Transpose to (H, M, D) for cache update
        q = q.transpose(0, 1)  # (H, M, D)
        k = k.transpose(0, 1)  # (H, M, D)
        v = v.transpose(0, 1)  # (H, M, D)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        cache_K_new = self.cache_K
        cache_V_new = self.cache_V

        # Attention scores: (H, M, D) @ (H, D, P+M) -> (H, M, P+M)
        scores = torch.matmul(q, cache_K_new.transpose(1, 2))
        scores_perturb = (scores + self.noise) / self.tau

        # Softmax - using torch.softmax for TVM compatibility
        # weights = torch.softmax(scores, dim=-1)
        scores_exp = torch.exp(scores)
        scores_exp_perturb = torch.exp(scores_perturb)

        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        scores_sum_perturb = torch.sum(scores_exp_perturb, dim=-1, keepdim=True)

        weights = scores_exp / scores_sum
        
        # Apply attention: (H, M, P+M) @ (H, P+M, D) -> (H, M, D)
        output = torch.matmul(weights, cache_V_new)
        weights_perturb = scores_exp_perturb / scores_sum_perturb
        c_out = torch.sum(weights_perturb, dim=1, keepdim=True)
        
        # Transpose back and reshape: (H, M, D) -> (M, H, D) -> (M, N)
        output = output.transpose(0, 1)  # (M, H, D)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output, c_out


        # c = torch.matmul(q, k_cache.permute(0, 2, 1))
        # c_perturb = (c + self.noise) / self.tau

        # c_exp = torch.exp(c)
        # c_exp_perturb = torch.exp(c_perturb)

        # c_sum = c_exp.sum(dim=2)
        # c_sum_perturb = c_exp_perturb.sum(dim=2)

        # c_div = c_exp / c_sum.unsqueeze(-1)
        # c_div_perturb = c_exp_perturb / c_sum_perturb.unsqueeze(-1)
        # c_out = c_div_perturb.sum(dim=1)

        # o = torch.matmul(c_div, v_cache)
        # o1 = o.permute(1, 0, 2)
        # o2 = o1.contiguous().view(self.M, self.N)
        # return o2, c_out


if __name__ == "__main__":
    import trinity

    M, H, D, P = 16, 32, 128, 1008
    N = H * D

    X = torch.randn((M, N))
    K_cache = torch.randn((H, P+M, D))
    V_cache = torch.randn((H, P+M, D))
    tau = torch.tensor(1.5)
    noise = torch.randn((H, M, P + M))

    model = KeyformerAttn(M, H, D, P, K_cache, V_cache, tau, noise)
    result = trinity.optimize(model, X, basename="keyformer", verbose=True, skip_frontend=False, backend_max_benchmarks=99999, backend_cuda_graph=False)
