"""Setup TVM environment by installing .pth file to site-packages."""
import os
import sys
import site


def main():
    """Install _tvm_setup.pth to site-packages."""
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    pth_src = os.path.join(backend_dir, "_tvm_setup.pth")

    # Get site-packages directory
    site_packages = site.getsitepackages()[0]
    pth_dst = os.path.join(site_packages, "_tvm_setup.pth")

    # Remove existing file/symlink if exists
    if os.path.exists(pth_dst) or os.path.islink(pth_dst):
        os.remove(pth_dst)

    # Create symlink
    os.symlink(pth_src, pth_dst)
    print(f"Installed: {pth_dst} -> {pth_src}")

    # Verify TVM can be imported
    tvm_python = os.path.join(backend_dir, "relax", "python")
    if tvm_python not in sys.path:
        sys.path.insert(0, tvm_python)

    try:
        import tvm
        print(f"TVM successfully imported from: {tvm.__file__}")
    except ImportError as e:
        print(f"Warning: TVM import failed: {e}")
        print("You may need to build TVM first:")
        print("  cd relax && CMAKE_ARGS='-DUSE_CUDA=ON -DUSE_CUBLAS=ON' uv pip install -e . -v")


if __name__ == "__main__":
    main()
