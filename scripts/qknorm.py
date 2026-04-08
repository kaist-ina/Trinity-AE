import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

import torch
import torch.nn as nn


class QKNormAttn(nn.Module):
    def __init__(self, M, H, D, P, cache_K, cache_V, device=None, dtype=None):
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

        self.q_proj = torch.randn(self.N, self.N, device=device, dtype=dtype)
        self.k_proj = torch.randn(self.N, self.N, device=device, dtype=dtype)
        self.v_proj = torch.randn(self.N, self.N, device=device, dtype=dtype)

        self.register_buffer("cache_K", cache_K.to(device))
        self.register_buffer("cache_V", cache_V.to(device))

    def forward(self, X):
        # q1 = self.q_proj(X)
        # k1 = self.k_proj(X)
        # v1 = self.v_proj(X)
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

        q_norm = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) / self.D)
        k_norm = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) / self.D)

        # q_norm = q / torch.sqrt((q * q).sum(dim=2) / self.D).unsqueeze(2)
        # k_norm = k / torch.sqrt((k * k).sum(dim=2) / self.D).unsqueeze(2)

        self.cache_K[:, self.P:self.P+self.M, :] = k_norm
        self.cache_V[:, self.P:self.P+self.M, :] = v
        cache_K_new = self.cache_K
        cache_V_new = self.cache_V

        # Transpose q to (H, M, D)

        # Attention scores: (H, M, D) @ (H, D, P+M) -> (H, M, P+M)
        scores = torch.matmul(q_norm, cache_K_new.transpose(1, 2))
        
        # Softmax - using torch.softmax for TVM compatibility
        # weights = torch.softmax(scores, dim=-1)
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum
        
        # Apply attention: (H, M, P+M) @ (H, P+M, D) -> (H, M, D)
        output = torch.matmul(weights, cache_V_new)
        
        # Transpose back and reshape: (H, M, D) -> (M, H, D) -> (M, N)
        output = output.transpose(0, 1)  # (M, H, D)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output


if __name__ == "__main__":
    import trinity

    M, H, D, P = 16, 32, 128, 1008
    N = H * D

    X = torch.randn((M, N))
    K_cache = torch.randn((H, P+M, D))
    V_cache = torch.randn((H, P+M, D))

    model = QKNormAttn(M, H, D, P, K_cache, V_cache)
    result = trinity.optimize(model, X, basename="qknorm", verbose=True, skip_frontend=True, backend_max_benchmarks=99999)
