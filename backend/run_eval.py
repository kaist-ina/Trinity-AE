from codegen.convert_module import convert_ir_to_triton
import argparse, torch, importlib.util, sys, json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--o", type=int, default=2, help="0 only convert, 1 only test, 2 both convert and test")
    parser.add_argument("--m", type=str, default="llama", help="Input model type")
    parser.add_argument("--t", type=str, default="vanilla", help="Benchmark type")
    parser.add_argument("--n", type=int, default=0, help="Case number for IR")
    parser.add_argument("--d", type=int, default=0, help="Type device number")
    parser.add_argument("--baseline", nargs="*", default=["trinity"], help="List of baselines")
    parser.add_argument("--print_output", action="store_true")
    args = parser.parse_args()

    num = args.n
    option = args.o
    model = args.m
    target = args.t
    baseline = args.baseline
    print_output = args.print_output
    
    device = torch.device(f'cuda:{args.d}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    dtype = torch.float16

    case_file = f"./results/{target}/{target}_{model}_case{num}.txt"
    output_file = f"./results/{target}/{target}_{model}_benchmark{num}.py"
    module_name = f"{target}_{model}_best"

    # Load model config from JSON
    with open('./model_configs.json', 'r') as f:
        model_configs = json.load(f)

    if model not in model_configs:
        raise ValueError(f"Unknown model: {model}. Available: {list(model_configs.keys())}")

    config = model_configs[model]
    M = config['M']
    D = config['D']
    N = config['N']
    P = config['P'] - M
    H = config['H']
    N4 = config['N4']
    num_group = config.get('num_group', 4)

    constants = {
        'M': M,
        'D': D,
        'N': N,
        'P': P,
        'H': H,
        'N4': N4,
    }
    
    tensor_shapes = {
        'X': ('M', 'N'),
        'X2': ('M',),
        'X_norm': ('M', 'N'),

        'WQ': ('N', 'N'),
        'WK': ('N', 'N'),
        'WV': ('N', 'N'),

        'Q1': ('M', 'N'),
        'K1': ('M', 'N'),
        'V1': ('M', 'N'),
        
        'Q2': ('M', 'H', 'D'),
        'K2': ('M', 'H', 'D'),
        'V2': ('M', 'H', 'D'),

        'K_cache': ('H', 'P+M', 'D'),
        'V_cache': ('H', 'P+M', 'D'),

        'Q': ('H', 'M', 'D'),
        'K': ('H', 'M', 'D'),
        'V': ('H', 'M', 'D'),

        'O': ('H', 'M', 'D'),
        'O1': ('M', 'H', 'D'),
        'O2': ('M', 'N'),

        'C': ('H', 'M', 'P+M'),
        'C_exp': ('H', 'M', 'P+M'),
        'C_div': ('H', 'M', 'P+M'),
        'C_sum': ('H', 'M'),
        'noise': ('H', 'M', 'P+M'),
        'C_perturb': ('H', 'M', 'P+M'),
        'C_exp_perturb': ('H', 'M', 'P+M'),
        'C_sum_perturb': ('H', 'M'),
        'C_div_perturb': ('H', 'M', 'P+M'),
        'C_out': ('H', 'P+M'),
        'C_out1': ('H', 'P+M'),
        'C_out2': ('H', 'P+M'),

        'Q_norm': ('H', 'M', 'D'),
        'K_norm': ('H', 'M', 'D'),

        'WO': ('N', 'N'),
        'attn_O1': ('M', 'N'),
        'attn_O2': ('M', 'N'),
        'attn_O3': ('M'),
        'attn_O_norm': ('M', 'N'),
        'WFF1a': ('N', 'N4'),
        'WFF1b': ('N', 'N4'),
        'FF1a': ('M', 'N4'),
        'FF1b': ('M', 'N4'),
        'FF1b_silu': ('M', 'N4'),
        'FF1': ('M', 'N4'),
        'FF2': ('M', 'N'),
        'WFF2': ('N4', 'N'),
    }

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # --------------- Init for Attention ---------------------
    std = 0.01
    X = torch.randn((M, N), device=device, dtype=dtype) * std
    X2 = torch.zeros((M,), device=device, dtype=dtype)
    X_norm = torch.zeros((M, N), device=device, dtype=dtype)

    WQ = torch.randn((N, N), device=device, dtype=dtype) * std
    WK = torch.randn((N, N), device=device, dtype=dtype) * std
    WV = torch.randn((N, N), device=device, dtype=dtype) * std

    Q1 = torch.zeros((M, N), device=device, dtype=dtype)
    K1 = torch.zeros((M, N), device=device, dtype=dtype)
    V1 = torch.zeros((M, N), device=device, dtype=dtype)

    Q2 = torch.zeros((M, H, D), device=device, dtype=dtype)
    K2 = torch.zeros((M, H, D), device=device, dtype=dtype)
    V2 = torch.zeros((M, H, D), device=device, dtype=dtype)

    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std

    K_cache_flashinfer = K_cache.clone().transpose(0, 1).contiguous()
    V_cache_flashinfer = V_cache.clone().transpose(0, 1).contiguous()

    Q = torch.zeros((H, M, D), device=device, dtype=dtype)
    K = torch.zeros((H, M, D), device=device, dtype=dtype)
    V = torch.zeros((H, M, D), device=device, dtype=dtype)

    O = torch.zeros((H, M, D), device=device, dtype=dtype)
    O1 = torch.zeros((M, H, D), device=device, dtype=dtype)
    O2 = torch.zeros((M, N), device=device, dtype=dtype) * std

    Q_norm = torch.zeros((H, M, D), device=device, dtype=dtype)
    K_norm = torch.zeros((H, M, D), device=device, dtype=dtype)

    # --------------- Additional init for RoCo ---------------------
    C_exp = torch.zeros((H, M, P+M), device=device, dtype=torch.float32)
    C_out1 = torch.zeros((H, P+M), device=device, dtype=dtype) * std
    C_out2 = torch.zeros((H, P+M), device=device, dtype=dtype) * std

    # --------------- Additional init for KeyFormer ---------------------
    C = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_sum = torch.zeros((H, M), device=device, dtype=dtype)
    C_sum_perturb = torch.zeros((H, M), device=device, dtype=dtype)
    C_exp_perturb = torch.zeros((H, M, P+M), device=device, dtype=torch.float32)
    C_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_out = torch.zeros((H, P+M), device=device, dtype=dtype)
    noise = torch.randn((H, M, P+M), device=device, dtype=dtype) * std

    # --------------- Init for FFN ---------------------
    std = 0.001
    if target == "ffn":
        O2 = O2
    else:
        O2 = torch.randn(M, N, dtype=dtype, device=device) * std
    FF1 = torch.zeros(M, N4, dtype=dtype, device=device)
    FF1a = torch.zeros(M, N4, dtype=dtype, device=device)
    FF1b = torch.zeros(M, N4, dtype=dtype, device=device)
    FF1b_silu = torch.zeros(M, N4, dtype=dtype, device=device)
    FF2 = torch.zeros(M, N4, dtype=dtype, device=device)
    WFF1a = torch.randn(N, N4, dtype=dtype, device=device) * std
    WFF1b = torch.randn(N, N4, dtype=dtype, device=device) * std
    WFF2 = torch.randn(N4, N, dtype=dtype, device=device) * std
    WO = torch.randn(N, N, dtype=dtype, device=device) * std
    attn_O1 = torch.zeros(M, N, dtype=dtype, device=device)
    attn_O2 = torch.zeros(M, N, dtype=dtype, device=device)
    attn_O3 = torch.zeros((M,), dtype=dtype, device=device)
    attn_O_norm = torch.zeros(M, N, dtype=dtype, device=device)



    out = O2.clone()
    ITER = 1000

    # Initialize timing variables
    trinity_time = None
    tensorrt_time = None
    torcheager_time = None
    inductor_time = None
    flashinfer_time = None
    flashtensor_time = None

    if option == 0:
        with open(case_file, "r") as f:
            ir = f.read().strip()
            triton_code = convert_ir_to_triton(ir, tensor_shapes, constants)

            with open(output_file, "w") as f:
                f.write(triton_code)
            
            print("="*50)
            print("Triton kernel generated successfully!")
        return

    match target:
        case "vanilla":
            from baselines import Vanilla, TensorRT_Vanilla, FlashInfer_Vanilla
            trt = TensorRT_Vanilla(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV, device, dtype)
            ti = Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
            fi = FlashInfer_Vanilla(M, N, D, P, K_cache_flashinfer.clone(), V_cache_flashinfer.clone(), WQ, WK, WV, device, dtype)
            from flashtensor.h100_vanilla import bench_vanilla
            ft = bench_vanilla
        case "prenorm":
            from baselines import PreNorm, TensorRT_PreNorm, FlashInfer_PreNorm
            trt = TensorRT_PreNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV, device, dtype)
            ti = PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
            fi = None
            from flashtensor.h100_prenorm import bench_prenorm
            ft = bench_prenorm
        case "keyformer":
            from baselines import KeyFormer, TensorRT_KeyFormer
            trt = TensorRT_KeyFormer(M, N, D, H, K_cache.clone(), V_cache.clone(), P, noise, WQ, WK, WV, device, dtype)
            ti = KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
            fi = None
            from flashtensor.h100_kf import bench_kf
            ft = bench_kf
        case "qknorm":
            from baselines import QKNorm, TensorRT_QKNorm, FlashInfer_QKNorm
            trt = TensorRT_QKNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV, device, dtype)
            ti = QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
            fi = None
            from flashtensor.h100_qknorm import bench_qknorm
            ft = bench_qknorm
        case "roco":
            from baselines import RoCo, TensorRT_RoCo, FlashInfer_RoCo
            trt = TensorRT_RoCo(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV, device, dtype)
            ti = RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV, device, dtype)
            fi = None
            from flashtensor.h100_roco import bench_roco
            ft = bench_roco
        case "ffn":
            from baselines import FFN, TensorRT_FFN
            trt = TensorRT_FFN(M, N, N4, WO=WO, WFF1a=WFF1a, WFF1b=WFF1b, WFF2=WFF2, device=device, dtype=dtype)
            ti = FFN(M, N, N4, WO=WO, WFF1a=WFF1a, WFF1b=WFF1b, WFF2=WFF2, device=device, dtype=dtype)
            fi = None
            ft = None

    # --------------- Trinity ---------------------
    print("="*50)
    print(f"Starting Trinity {target}...")
    if len(baseline) == 0 or "trinity" in baseline:
        if option == 0 or option == 2:
            # Convert IR to Triton kernel
            with open(case_file, "r") as f:
                ir = f.read().strip()
            triton_code = convert_ir_to_triton(ir, tensor_shapes, constants)

            with open(output_file, "w") as f:
                f.write(triton_code)
            
            print("="*50)
            print("Triton kernel generated successfully!")
        if option == 0:
            return

        spec = importlib.util.spec_from_file_location(module_name, output_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        forward = getattr(module, "forward")

        tensor_params = getattr(module, 'TENSOR_PARAMS')
        block_params = getattr(module, 'BLOCK_PARAMS')
        C_div = torch.zeros((H, M, P+M), device=device, dtype=dtype)
        C_div_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
        tensors = {
            'X': X, 'X2': X2, 'X_norm': X_norm,
            'WQ': WQ, 'WK': WK, 'WV': WV,
            'Q1': Q1, 'K1': K1, 'V1': V1,
            'Q2': Q2, 'K2': K2, 'V2': V2,
            'K_cache': K_cache, 'V_cache': V_cache,
            'Q': Q, 'K': K, 'V': V,
            'O': O, 'O1': O1, 'O2': O2,
            'C': C, 'C_exp': C_exp, 'C_div': C_div, 'C_sum': C_sum,
            'noise': noise, 'C_perturb': C_perturb,
            'C_exp_perturb': C_exp_perturb, 'C_sum_perturb': C_sum_perturb,
            'C_div_perturb': C_div_perturb,
            'C_out': C_out, 'C_out1': C_out1, 'C_out2': C_out2,
            'Q_norm': Q_norm, 'K_norm': K_norm,
            'WO': WO, 'attn_O1': attn_O1, 'attn_O2': attn_O2,
            'attn_O3': attn_O3, 'attn_O_norm': attn_O_norm,
            'WFF1a': WFF1a, 'WFF1b': WFF1b,
            'FF1a': FF1a, 'FF1b': FF1b, 'FF1b_silu': FF1b_silu,
            'FF1': FF1, 'FF2': FF2, 'WFF2': WFF2,
        }
        blocks = {
            'block_k': 0, 'block_n': 0, 'block_p': 0
        }
        args = []
        for param in tensor_params:
            if param in tensors:
                args.append(tensors[param])
            else:
                raise ValueError(f"Unknown tensor parameter: {param}")
        for param in block_params:
            if param in blocks:
                args.append(blocks[param])
            else:
                raise ValueError(f"Unknown block parameter: {param}")

        for _ in range(10):
            forward(*args)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(ITER):
            forward(*args)
        end_event.record()
        torch.cuda.synchronize()

        trinity_time = start_event.elapsed_time(end_event) / ITER

        if print_output:
            print(O2)

    # ----------------- TensorRT ---------------------
    if len(baseline) == 0 or "tensorrt" in baseline:
        print("="*50)
        print(f"Starting TensorRT {target}...")

        if target == "ffn":
            inputs = (O2, X)
        else:
            inputs = (X,)

        trt.half()
        
        with torch.no_grad():
            for _ in range(10):
                out = trt(*inputs)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                _ = trt(*inputs)
            end_event.record()
            torch.cuda.synchronize()

            tensorrt_time = start_event.elapsed_time(end_event) / ITER

        if print_output:
            print(out)

    # ----------------- Pytorch Eager ----------------------
    if len(baseline) == 0 or "pytorch" in baseline:
        print("="*50)
        print(f"Starting Pytorch Eager {target}...")
        
        if target == "ffn":
            inputs = (O2, X)
        else:
            inputs = (X,)

        ti = ti.eval()
        
        with torch.no_grad():
            for _ in range(10):
                out = ti(*inputs)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                _ = ti(*inputs)
            end_event.record()
            torch.cuda.synchronize()

            torcheager_time = start_event.elapsed_time(end_event) / ITER

        if print_output:
            print(out)

    # ----------------- Torch Inductor ---------------------
    if len(baseline) == 0 or "inductor" in baseline:
        print("="*50)
        print(f"Starting Torch Inductor {target}...")

        if target == "ffn":
            inputs = (O2, X)
        else:
            inputs = (X,)

        mode = "max-autotune-no-cudagraphs"
        compiled_model = torch.compile(ti, backend="inductor", mode=mode, fullgraph=True)
        for _ in range(10):
            out = compiled_model(*inputs)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(ITER):
            _ = compiled_model(*inputs)
        end_event.record()
        torch.cuda.synchronize()

        inductor_time = start_event.elapsed_time(end_event) / ITER

        if print_output:
            print(out)
    
    # ----------------- FlashInfer ---------------------
    if not fi is None and len(baseline) == 0 or "flashinfer" in baseline:
        print("="*50)
        print(f"Starting FlashInfer {target}...")
        
        fi.half()
        fi = fi.eval()
        
        with torch.no_grad():
            for _ in range(10):
                out = fi(X)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(ITER):
                _ = fi(X)
            end_event.record()
            torch.cuda.synchronize()

            flashinfer_time = start_event.elapsed_time(end_event) / ITER

        if print_output:
            print(out)
    
    # ----------------- FlashTensor ---------------------
    if not ft is None and len(baseline) == 0 or "flashtensor" in baseline:
        print("="*50)
        print(f"Starting FlashTensor {target}...")


        flashtensor_time = ft(model, M, N, P, D, H, device, dtype)

    # ----------------- Results Summary ---------------------
    results = []
    if trinity_time is not None:
        results.append(("Trinity", trinity_time))
    if tensorrt_time is not None:
        results.append(("TensorRT", tensorrt_time))
    if torcheager_time is not None:
        results.append(("PyTorch Eager", torcheager_time))
    if inductor_time is not None:
        results.append(("Torch Inductor", inductor_time))
    if flashinfer_time is not None:
        results.append(("FlashInfer", flashinfer_time))
    if flashtensor_time is not None:
        results.append(("FlashTensor", flashtensor_time))

    if results:
        print("\n" + "="*50)
        print(f"BENCHMARK RESULTS: {target.upper()} ({model.upper()})")
        print("="*50)

        if trinity_time is not None:
            print(f"{'Method':<20} {'Time (ms)':<15} {'Speedup':<10}")
            print("-" * 45)
            for name, time_val in results:
                speedup = time_val / trinity_time
                speedup_str = f"{speedup:.2f}x" if name != "Trinity" else "1.00x"
                print(f"{name:<20} {time_val:<15.4f} {speedup_str:<10}")
        else:
            print(f"{'Method':<20} {'Time (ms)':<15}")
            print("-" * 35)
            for name, time_val in results:
                print(f"{name:<20} {time_val:<15.4f}")


if __name__ == "__main__":
    main()
