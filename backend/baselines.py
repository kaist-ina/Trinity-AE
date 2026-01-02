import torch.nn as nn
import torch.nn.functional as F
import torch
import flashinfer
import math
import tensorrt as trt
import tempfile
import os

class Vanilla(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):

        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
    
        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # q = q.transpose(0, 1)

        # scores = torch.matmul(q, self.cache_K.transpose(1, 2))
        # weights = F.softmax(scores, dim=-1)
        
        # output = torch.matmul(weights, self.cache_V)
        # output = output.view(self.M, self.H * self.D)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum

        # weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        # output = output.reshape(self.M, self.H * self.D)
        return output

class PreNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)

        # q = torch.matmul(X_norm, self.W_q)
        # k = torch.matmul(X_norm, self.W_k)
        # v = torch.matmul(X_norm, self.W_v)

        qkv = torch.matmul(X_norm, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        # output = output.reshape(self.M, self.H * self.D)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output

class KeyFormer(nn.Module):
    def __init__(self, M, N, D, P, noise, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.noise = noise.to(device=device, dtype=dtype)

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)
        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V


        q = q.transpose(0, 1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        perturb = (scores + self.noise) / 1.5
        # weights = F.softmax(scores, dim=-1)

        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum

        perturb_exp = torch.exp(perturb)
        perturb_sum = torch.sum(perturb_exp, dim=-1, keepdim=True)
        perturb_out = perturb_exp / perturb_sum

        # perturb_out = F.softmax(perturb, dim=-1)
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output, perturb_out

class QKNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        q_var = q.pow(2).mean(-1, keepdim=True)
        k_var = k.pow(2).mean(-1, keepdim=True)
        q_norm = q * torch.rsqrt(q_var)
        k_norm = k * torch.rsqrt(k_var)

        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k_norm], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)
        self.cache_K[:, self.P:self.P+self.M, :] = k_norm
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        # scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
        # weights = F.softmax(scores, dim=-1)
        scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output

class RoCo(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        # q = torch.matmul(X, self.W_q)
        # k = torch.matmul(X, self.W_k)
        # v = torch.matmul(X, self.W_v)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
        # v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        # scores = torch.matmul(q, k_cache.transpose(1, 2))
        # weights = F.softmax(scores, dim=-1)

        scores = torch.matmul(q, k_cache.transpose(1, 2))
        scores_exp = torch.exp(scores)
        scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
        weights = scores_exp / scores_sum

        weights_sum = weights.sum(dim=1)
        weights_sqr_sum = weights.pow(2).sum(dim=1)
        
        output = torch.matmul(weights, v_cache)
        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)

        return output, weights_sum, weights_sqr_sum

class FFN(nn.Module):
    def __init__(self, M, N, N4, WO=None, WFF1a=None, WFF1b=None, WFF2=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.N4 = N4
        self.device = device
        self.dtype = dtype
        self.WO = WO.to(device=device, dtype=dtype)
        self.WFF1a = WFF1a.to(device=device, dtype=dtype)
        self.WFF1b = WFF1b.to(device=device, dtype=dtype)
        self.WFF2 = WFF2.to(device=device, dtype=dtype)
    
    def forward(self, O2, X):
        attn_O1 = torch.matmul(O2, self.WO)
        attn_O2 = attn_O1 + X
        attn_O3 = attn_O2.pow(2).mean(-1, keepdim=True)
        attn_O_norm = attn_O2 * torch.rsqrt(attn_O3)

        FF1a = torch.matmul(attn_O_norm, self.WFF1a)
        FF1b = torch.matmul(attn_O_norm, self.WFF1b)
        FF1b_silu = FF1b * torch.sigmoid(FF1b)

        FF1 = FF1a * FF1b_silu
        FF2 = torch.matmul(FF1, self.WFF2)
        
        return FF2

class SimpleAttention(nn.Module):
    """Simple manual attention implementation for comparison"""
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)
    
    def forward(self, X):
        seq_len  = X.shape[0]
        
        # variance = X.pow(2).mean(-1, keepdim=True)
        # X_norm = X * torch.rsqrt(variance)

        q = torch.matmul(X, self.W_q)
        k = torch.matmul(X, self.W_k)
        v = torch.matmul(X, self.W_v)

        q = q.view(seq_len, self.H, self.D)
        k = k.view(seq_len, self.H, self.D)
        v = v.view(seq_len, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        self.cache_K[:, self.P:self.P+seq_len, :] = k
        self.cache_V[:, self.P:self.P+seq_len, :] = v

        q = q.transpose(0, 1)

        scores = torch.matmul(q, self.cache_K.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(weights, self.cache_V)
        output = output.view(seq_len, self.H * self.D)

        return output

class TensorRT_Vanilla(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P  # Add P for cache length
        self.device = device
        self.dtype = dtype
        self.output = torch.empty(self.M, self.N, dtype=dtype, device=device)

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                qkv = torch.matmul(X, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                cache_K[:, self.P:self.P+seq_len, :] = k
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        device = self.device
        dtype = self.dtype

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            model = model.to(device)

            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output'],
                opset_version=18,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            if not parser.parse_from_file(onnx_path):
                raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            self.output.data_ptr()
        ]

        # Execute TensorRT engine
        _ = self.context.execute_v2(bindings)
        
        return self.output

class TensorRT_PreNorm(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P
        self.device = device
        self.dtype = dtype
        self.output = torch.empty(self.M, self.N, dtype=dtype, device=device)

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                variance = X.pow(2).mean(-1, keepdim=True)
                X_norm = X * torch.rsqrt(variance)

                qkv = torch.matmul(X_norm, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                cache_K[:, self.P:self.P+seq_len, :] = k
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        device = self.device
        dtype = self.dtype

        try:
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            model = model.to(device)

            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output'],
                opset_version=18,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            if not parser.parse_from_file(onnx_path):
                raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            self.output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return self.output

class TensorRT_KeyFormer(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, noise, W_q, W_k, W_v, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P
        self.device = device
        self.dtype = dtype
        self.output = torch.empty(self.M, self.N, dtype=dtype, device=device)
        self.perturb_output = torch.empty(self.H, self.P+self.M, dtype=dtype, device=device)

        self.noise = noise.to(device=device, dtype=dtype)
        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, noise, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P

                self.noise = noise
                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # q = torch.matmul(X, self.W_q)
                # k = torch.matmul(X, self.W_k)
                # v = torch.matmul(X, self.W_v)
                qkv = torch.matmul(X, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                # k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)
                cache_K[:, self.P:self.P+seq_len, :] = k
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                perturb = (scores+self.noise) / 1.5

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                perturb_exp = torch.exp(perturb)
                perturb_sum = torch.sum(perturb_exp, dim=-1, keepdim=True)
                perturb_out = perturb_exp / perturb_sum

                # weights = F.softmax(scores, dim=-1)
                # perturb_out = F.softmax(perturb, dim=-1)

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)

                return output, perturb_out

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        device = self.device
        dtype = self.dtype

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.noise,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output', 'perturb_out'],
                opset_version=18,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            if not parser.parse_from_file(onnx_path):
                raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            self.output.data_ptr(),
            self.perturb_output.data_ptr()
        ]

        # Execute TensorRT engine
        _ = self.context.execute_v2(bindings)
        
        return self.output

class TensorRT_QKNorm(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P
        self.device = device
        self.dtype = dtype
        self.output = torch.empty(self.M, self.N, dtype=dtype, device=device)

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # q = torch.matmul(X, self.W_q)
                # k = torch.matmul(X, self.W_k)
                # v = torch.matmul(X, self.W_v)
                qkv = torch.matmul(X, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                q_var = q.pow(2).mean(-1, keepdim=True)
                k_var = k.pow(2).mean(-1, keepdim=True)
                q_norm = q * torch.rsqrt(q_var)
                k_norm = k * torch.rsqrt(k_var)

                # k_cache = torch.cat([cache_K[:, :self.P, :], k_norm], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)

                cache_K[:, self.P:self.P+seq_len, :] = k_norm
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V

                scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)

                # scores = torch.matmul(q_norm, k_cache.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)

                # output = torch.matmul(weights, v_cache)
                # output = output.view(seq_len, self.H*self.D)

                return output

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        device = self.device
        dtype = self.dtype

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)

            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output'],
                opset_version=18,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            if not parser.parse_from_file(onnx_path):
                raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            self.output.data_ptr()
        ]

        # Execute TensorRT engine
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return self.output

class TensorRT_RoCo(nn.Module):
    def __init__(self, M, N, D, H, cache_K, cache_V, P, W_q, W_k, W_v, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.H = H
        self.P = P
        self.device = device
        self.dtype = dtype
        self.output = torch.empty(self.M, self.N, dtype=dtype, device=device)
        self.weights_sum = torch.empty(self.H, self.P+self.M, dtype=dtype, device=device)
        self.weights_sqr_sum = torch.empty(self.H, self.P+self.M, dtype=dtype, device=device)

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class AttentionModel(nn.Module):
            def __init__(self, N, D, H, P, W_q, W_k, W_v):
                super().__init__()
                self.N = N
                self.D = D
                self.H = H
                self.P = P
                # self.weight = nn.Parameter(torch.ones(H, dtype=dtype, device=device))
                # self.register_buffer('W_q', W_q)
                # self.register_buffer('W_k', W_k)
                # self.register_buffer('W_v', W_v)

                self.W_q = W_q
                self.W_k = W_k
                self.W_v = W_v
                self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1)

            def forward(self, X, cache_K, cache_V):
                seq_len = X.shape[0]

                # q = torch.matmul(X, self.W_q)
                # k = torch.matmul(X, self.W_k)
                # v = torch.matmul(X, self.W_v)

                qkv = torch.matmul(X, self.W_qkv)
                q, k, v = torch.chunk(qkv, 3, dim=-1)

                q = q.view(seq_len, self.H, self.D)
                k = k.view(seq_len, self.H, self.D)
                v = v.view(seq_len, self.H, self.D)

                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)

                # k_cache = torch.cat([cache_K[:, :self.P, :], k], dim=1)
                # v_cache = torch.cat([cache_V[:, :self.P, :], v], dim=1)
                cache_K[:, self.P:self.P+seq_len, :] = k
                cache_V[:, self.P:self.P+seq_len, :] = v
                k_cache = cache_K
                v_cache = cache_V

                # scores = torch.matmul(q, k_cache.transpose(1, 2))
                # weights = F.softmax(scores, dim=-1)

                scores = torch.matmul(q, k_cache.transpose(1, 2))
                scores_exp = torch.exp(scores)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                weights = scores_exp / scores_sum

                weights_sum = weights.sum(dim=1)
                weights_sqr_sum = weights.pow(2).sum(dim=1)

                # output = torch.matmul(weights, v_cache)
                # output = output.view(seq_len, self.H*self.D)

                output = torch.matmul(weights, v_cache)
                output = output.permute(1, 0, 2)
                output = output.contiguous().view(seq_len, self.H * self.D)

                return output, weights_sum, weights_sqr_sum

        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        device = self.device
        dtype = self.dtype

        try:
            # Create model with proper scale and weights
            model = AttentionModel(
                self.N,
                self.D,
                self.H,
                self.P,
                self.W_q,
                self.W_k,
                self.W_v
            )
            # Copy weights from parent model
            # model.weight.data = self.weight.data.clone()
            model = model.to(device)
            
            # Dummy inputs for export
            seq_len = self.M
            dummy_X = torch.randn(seq_len, self.N, dtype=dtype, device=device)
            dummy_cache_K = self.cache_K.clone().to(device)
            dummy_cache_V = self.cache_V.clone().to(device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_X, dummy_cache_K, dummy_cache_V),
                onnx_path,
                input_names=['X', 'cache_K', 'cache_V'],
                output_names=['output', 'weights_sum', 'weights_sqr_sum'],
                opset_version=18,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            if not parser.parse_from_file(onnx_path):
                raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16
            config.set_flag(trt.BuilderFlag.FP16)
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def forward(self, X):
        # Create bindings - pass X directly to TensorRT
        bindings = [
            X.data_ptr(),
            self.cache_K.data_ptr(),
            self.cache_V.data_ptr(),
            self.output.data_ptr(),
            self.weights_sum.data_ptr(),
            self.weights_sqr_sum.data_ptr()
        ]

        # Execute TensorRT engine
        _ = self.context.execute_v2(bindings)
        
        return self.output

class TensorRT_FFN(nn.Module):
    def __init__(self, M, N, N4, WO=None, WFF1a=None, WFF1b=None, WFF2=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.N4 = N4
        self.device = device
        self.dtype = dtype
        self.WO = WO.to(device=device, dtype=dtype) if WO is not None else None
        self.WFF1a = WFF1a.to(device=device, dtype=dtype) if WFF1a is not None else None
        self.WFF1b = WFF1b.to(device=device, dtype=dtype) if WFF1b is not None else None
        self.WFF2 = WFF2.to(device=device, dtype=dtype) if WFF2 is not None else None

        self.output = torch.empty((self.M, self.N), dtype=dtype, device=device)

        self.engine = None
        self.context = None
        self.build_engine()
    
    def build_engine(self):
        class TensorOpsModel(nn.Module):
            def __init__(self, N, WO, WFF1a, WFF1b, WFF2):
                super().__init__()
                self.N = N
                self.WO = WO
                self.WFF1a = WFF1a
                self.WFF1b = WFF1b
                self.WFF2 = WFF2
            
            def forward(self, O2, X):
                attn_O1 = torch.matmul(O2, self.WO)
                attn_O2 = attn_O1 + X
                attn_O3 = attn_O2.pow(2).mean(-1, keepdim=True)
                attn_O_norm = attn_O2 * torch.rsqrt(attn_O3)

                FF1a = torch.matmul(attn_O_norm, self.WFF1a)
                FF1b = torch.matmul(attn_O_norm, self.WFF1b)
                FF1b_silu = FF1b * torch.sigmoid(FF1b)

                FF1 = FF1a * FF1b_silu
                FF2 = torch.matmul(FF1, self.WFF2)
                
                return FF2
        
        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()

        device = self.device
        dtype = self.dtype

        try:
            model = TensorOpsModel(self.N, self.WO, self.WFF1a, self.WFF1b, self.WFF2)

            dummy_O2 = torch.randn(self.M, self.N, dtype=dtype, device=device)
            dummy_X = torch.randn(self.M, self.N, dtype=dtype, device=device)

            torch.onnx.export(
                model,
                (dummy_O2, dummy_X),
                onnx_path,
                input_names=['O2','X'],
                output_names=['FF2'],
                opset_version=18,
                do_constant_folding=True,
            )

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)

            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None: 
                raise RuntimeError("Failed to build TensorRT engine")

            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorrRT engine")
            
            self.context = self.engine.create_execution_context()
        
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
    
    def forward(self, O2, X):
        bindings = [
            O2.data_ptr(),
            X.data_ptr(),
            self.output.data_ptr()
        ]

        self.context.execute_v2(bindings)
        return self.output


class FlashInfer_Vanilla(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        # QKV projection
        # q = torch.matmul(X, self.W_q).view(self.M, self.H, self.D)
        # k = torch.matmul(X, self.W_k).view(self.M, self.H, self.D)
        # v = torch.matmul(X, self.W_v).view(self.M, self.H, self.D)

        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
    
        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        self.cache_K[self.P:self.P+self.M, :, :] = k
        self.cache_V[self.P:self.P+self.M, :, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="NHD",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )
        output = output.contiguous().view(self.M, self.H * self.D)
        return output

class FlashInfer_PreNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        variance = X.pow(2).mean(-1, keepdim=True)
        X_norm = X * torch.rsqrt(variance)

        qkv = torch.matmul(X_norm, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
    
        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        self.cache_K[self.P:self.P+self.M, :, :] = k
        self.cache_V[self.P:self.P+self.M, :, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="NHD",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )

        output = output.contiguous().view(self.M, self.H * self.D)
        return output

class FlashInfer_KeyFormer(nn.Module):
    def __init__(self, M, N, D, P, noise, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.noise = noise.to(device=device, dtype=dtype)

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
    
        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )
        
        scores = torch.matmul(q.transpose(0, 1), k_cache.transpose(1, 2))
        perturb = (scores + self.noise) / 1.5
        weights = F.softmax(scores, dim=-1)
        perturb_out = F.softmax(perturb, dim=-1)

        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)
        return output

class FlashInfer_QKNorm(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
    
        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)

        # Normalize Q and K in (H, M, D) format
        q_var = q.pow(2).mean(-1, keepdim=True)
        k_var = k.pow(2).mean(-1, keepdim=True)
        q_norm = q * torch.rsqrt(q_var)
        k_norm = k * torch.rsqrt(k_var)

        # Transpose Q back to (M, H, D) for flashinfer
        self.cache_K[self.P:self.P+self.M, :, :] = k_norm
        self.cache_V[self.P:self.P+self.M, :, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        output = flashinfer.single_prefill_with_kv_cache(
            q=q_norm,
            k=k_cache,
            v=v_cache,
            kv_layout="NHD",
            pos_encoding_mode="NONE",
            sm_scale=1.0  # Important: use 1.0 since Q and K are already normalized
        )

        output = output.contiguous().view(self.M, self.H * self.D)
        return output

class FlashInfer_RoCo(nn.Module):
    def __init__(self, M, N, D, P, cache_K, cache_V, W_q=None, W_k=None, W_v=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.D = D
        self.P = P
        self.H = N // D
        self.device = device
        self.dtype = dtype

        self.W_q = W_q.to(device=device, dtype=dtype)
        self.W_k = W_k.to(device=device, dtype=dtype)
        self.W_v = W_v.to(device=device, dtype=dtype)
        self.W_qkv = torch.cat([self.W_q, self.W_k, self.W_v], dim=1).to(device=device, dtype=dtype)

        self.cache_K = cache_K.to(device)
        self.cache_V = cache_V.to(device)

    def forward(self, X):
        qkv = torch.matmul(X, self.W_qkv)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
    
        q = q.view(self.M, self.H, self.D)
        k = k.view(self.M, self.H, self.D)
        v = v.view(self.M, self.H, self.D)
        
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        self.cache_K[:, self.P:self.P+self.M, :] = k
        self.cache_V[:, self.P:self.P+self.M, :] = v
        k_cache = self.cache_K
        v_cache = self.cache_V

        output = flashinfer.single_prefill_with_kv_cache(
            q=q,
            k=k_cache,
            v=v_cache,
            kv_layout="HND",
            pos_encoding_mode="NONE",
            sm_scale=1.0
        )

        scores = torch.matmul(q.transpose(0, 1), k_cache.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)
        weights_sum = weights.sum(dim=1)
        weights_sqr_sum = weights.pow(2).sum(dim=1)

        output = output.permute(1, 0, 2)
        output = output.contiguous().view(self.M, self.H * self.D)
        return output
