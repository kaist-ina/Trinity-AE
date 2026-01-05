"""Install all dependencies including TVM Relax and Mirage with proper build options."""
import os
import subprocess
import sys
import site
import re

def get_cuda_architectures():
    """Detect GPU compute capabilities and return CMAKE_CUDA_ARCHITECTURES string."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        caps = set()
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                # Convert "9.0" -> "90", "12.0" -> "120"
                major, minor = line.strip().split(".")
                caps.add(f"{major}{minor}")

        if caps:
            arch_str = ";".join(sorted(caps))
            print(f"Detected GPU architectures: {arch_str}")
            return arch_str
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Warning: Could not detect GPU architecture: {e}")

    # Fallback to native (let CMake detect at build time)
    print("Using native GPU architecture detection")
    return "native"

def _detect_cuda_major() -> int | None:
    try:
        out = subprocess.run(
            ["/usr/local/cuda/bin/nvcc", "--version"],
            capture_output=True, text=True, check=True
        ).stdout
        m = re.search(r"release\s+(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, check=True
        ).stdout
        m = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass

    return None


def choose_trt_extra() -> str | None:
    override = os.environ.get("TRINITY_TRT_EXTRA", "").strip()
    if override:
        return None if override.lower() == "none" else override

    major = _detect_cuda_major()
    if major == 12:
        return "trt-cu12"
    if major == 13:
        return "trt-cu13"
    return None

def main():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    relax_dir = os.path.join(backend_dir, "relax")
    mirage_dir = os.path.join(backend_dir, "mirage_eval", "src", "mirage")
    venv_dir = os.path.join(backend_dir, ".venv")
    
    common_env = os.environ.copy()
    common_env["UV_PROJECT_ENVIRONMENT"] = venv_dir
    
    trt_extra = choose_trt_extra()

    # 0. Sync main package dependencies first (before building TVM/Mirage)
    print("=" * 50)
    print("Syncing main package dependencies...")
    print("=" * 50)
    
    sync_cmd = ["uv", "sync"]
    if trt_extra:
        print(f"Enabling optional dependency: {trt_extra}")
        sync_cmd += ["--extra", trt_extra]
        
    subprocess.run(
        sync_cmd,
        cwd=backend_dir,
        env=common_env,
        check=True,
    )

    # 1. Initialize git submodules for TVM
    print("=" * 50)
    print("Initializing git submodules...")
    print("=" * 50)
    subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=relax_dir,
        check=True,
    )

    # Detect GPU architectures
    cuda_archs = get_cuda_architectures()

    # Build environment for TVM Relax
    env = os.environ.copy()
    cuda_home = "/usr/local/cuda"
    env["CMAKE_ARGS"] = (
        "-DUSE_CUDA=ON -DUSE_CUBLAS=ON -DUSE_CUTLASS=ON "
        f"-DCUDAToolkit_ROOT={cuda_home} "
        f"-DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc "
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}"
    )
    env["CUDA_HOME"] = cuda_home
    env["CUDAToolkit_ROOT"] = cuda_home
    env["PATH"] = cuda_home + "/bin:" + env.get("PATH", "")

    # 2. Build TVM Relax
    print("=" * 50)
    print("Building TVM Relax...")
    print("=" * 50)
    subprocess.run(
        ["uv", "pip", "install", "-e", ".", "-v"],
        cwd=relax_dir,
        env=env,
        check=True,
    )

    # 3. Build Mirage
    print("=" * 50)
    print("Building Mirage...")
    print("=" * 50)
    subprocess.run(
        ["uv", "pip", "install", "-e", "."],
        cwd=mirage_dir,
        check=True,
    )

    # 4. Install TVM .pth file
    print("=" * 50)
    print("Setting up TVM environment...")
    print("=" * 50)
    pth_src = os.path.join(backend_dir, "_tvm_setup.pth")
    # Use the virtual environment's site-packages
    venv_dir = os.path.join(backend_dir, ".venv")
    site_packages = None
    for lib_dir in ["lib", "lib64"]:
        lib_path = os.path.join(venv_dir, lib_dir)
        if os.path.isdir(lib_path):
            for entry in os.listdir(lib_path):
                if entry.startswith("python"):
                    candidate = os.path.join(lib_path, entry, "site-packages")
                    if os.path.isdir(candidate):
                        site_packages = candidate
                        break
        if site_packages:
            break
    if not site_packages:
        # Fallback to system site-packages
        site_packages = site.getsitepackages()[0]
    pth_dst = os.path.join(site_packages, "_tvm_setup.pth")

    if os.path.exists(pth_dst) or os.path.islink(pth_dst):
        os.remove(pth_dst)
    os.symlink(pth_src, pth_dst)
    print(f"Installed: {pth_dst}")

    # 5. Verify using the virtual environment's Python
    print("=" * 50)
    print("Verifying installation...")
    print("=" * 50)

    venv_python = os.path.join(venv_dir, "bin", "python")
    verify_script = """
import tvm
print(f"TVM: {tvm.__file__}")
import mirage
print(f"Mirage: {mirage.__file__}")
"""
    result = subprocess.run(
        [venv_python, "-c", verify_script],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Verification failed: {result.stderr}")
        return 1

    print("=" * 50)
    print("Installation complete!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
