import torch
import torch.nn as nn
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))



class Roco(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        # self.q_proj = nn.Linear(N, N, bias=False)
        # self.k_proj = nn.Linear(N, N, bias=False)
        # self.v_proj = nn.Linear(N, N, bias=False)

        self.q_proj = torch.randn(N, N, device=device, dtype=dtype)
        self.k_proj = torch.randn(N, N, device=device, dtype=dtype)
        self.v_proj = torch.randn(N, N, device=device, dtype=dtype)

        # cache는 buffer로 등록
        self.register_buffer('cache_K', cache_K.to(device))
        self.register_buffer('cache_V', cache_V.to(device))
    
    def forward(self, X):
        # X shape: (M, N) where M=16 (seq), N=4096 (hidden)
        # Project Q, K, V separately
        # q = self.q_proj(X)
        # k = self.k_proj(X)
        # v = self.v_proj(X)
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

        # Update cache - using slicing to avoid in-place operation issues
        # cache_K_new = self.cache_K.clone()
        # cache_V_new = self.cache_V.clone()
        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        cache_K_new = self.cache_K
        cache_V_new = self.cache_V

        # Transpose q to (H, M, D)

        # Attention scores: (H, M, D) @ (H, D, P+M) -> (H, M, P+M)
        scores = torch.matmul(q, cache_K_new.transpose(1, 2))
        
        # Softmax - using torch.softmax for TVM compatibility
        # weights = torch.softmax(scores, dim=-1)
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum
        
        # Apply attention: (H, M, P+M) @ (H, P+M, D) -> (H, M, D)
        output = torch.matmul(weights, cache_V_new)
        c_out1 = torch.sum(weights, dim=1, keepdim=True)
        c_out2 = torch.sum(weights*weights, dim=1, keepdim=True)
        
        # Transpose back and reshape: (H, M, D) -> (M, H, D) -> (M, N)
        output = output.transpose(0, 1)  # (M, H, D)
        output = output.contiguous().view(self.M, self.H * self.D)


        return output, c_out1, c_out2

if __name__ == "__main__":
    import trinity

    M, N, D, H, P = 16, 4096, 128, 32, 1008

    X = torch.randn((M, N))
    K_cache = torch.randn((H, P + M, D))
    V_cache = torch.randn((H, P + M, D))

    model = Roco(M, N, D, P, K_cache, V_cache)
    result = trinity.optimize(model, X, basename="roco", backend_max_benchmarks=9999, skip_frontend=True, device=1 ,verbose=True, backend_timeout_s=86400)

