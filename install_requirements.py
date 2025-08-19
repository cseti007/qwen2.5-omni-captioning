#!/usr/bin/env python3
"""
Install script for qwen2.5-omni-captioning requirements.
This script ensures proper installation order to avoid flash-attn build issues.
"""

import subprocess
import sys
import os

def run_pip_install(packages, description=""):
    """Run pip install with error handling"""
    if isinstance(packages, str):
        packages = [packages]
    
    print(f"\n{'='*50}")
    print(f"Installing {description}...")
    print(f"Packages: {', '.join(packages)}")
    print(f"{'='*50}")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        subprocess.check_call(cmd)
        print(f"✅ Successfully installed {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {description}: {e}")
        return False

def main():
    print("🚀 Starting qwen2.5-omni-captioning requirements installation...")
    
    # Step 1: Install core dependencies first (especially PyTorch and wheel)
    core_deps = [
        "wheel",  # Required for building packages
        "torch>=2.0.0",
        "transformers>=4.52.3",
        "accelerate",
        "toml"
    ]
    
    if not run_pip_install(core_deps, "core dependencies (PyTorch, transformers, etc.)"):
        sys.exit(1)
    
    # Step 2: Install VLLM and Qwen utilities
    vllm_deps = [
        "vllm>=0.8.5",
        "qwen-omni-utils[decord]"
    ]
    
    if not run_pip_install(vllm_deps, "VLLM and Qwen utilities"):
        sys.exit(1)
    
    # Step 3: Install other dependencies
    other_deps = [
        "gradio>=4.0.0",
        "opencv-python",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "setuptools_scm",
        "torchdiffeq",
        "resampy",
        "x_transformers",
        "pandas",
        "pathlib2",
        "tqdm"
    ]
    
    if not run_pip_install(other_deps, "other dependencies"):
        sys.exit(1)
    
    # Step 4: Install flash-attn separately with --no-build-isolation
    print(f"\n{'='*50}")
    print("Installing flash-attn (this may take a while)...")
    print("Note: flash-attn requires a compatible CUDA GPU and may fail on CPU-only systems")
    print("Using --no-build-isolation to use existing torch installation")
    print(f"{'='*50}")
    
    # Try with --no-build-isolation flag to use existing torch
    try:
        cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation", "flash-attn>=2.0.0"]
        subprocess.check_call(cmd)
        print("✅ Successfully installed flash-attn")
        flash_attn_success = True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install flash-attn with --no-build-isolation: {e}")
        print("Trying standard installation as fallback...")
        flash_attn_success = run_pip_install("flash-attn>=2.0.0", "flash-attn")
    
    if not flash_attn_success:
        print("\n⚠️  flash-attn installation failed!")
        print("This is often expected on CPU-only systems or incompatible GPUs.")
        print("The application should still work without flash-attn, but performance may be slower.")
        
        response = input("Continue without flash-attn? (y/n): ").lower().strip()
        if response != 'y':
            sys.exit(1)
    
    print("\n🎉 Installation completed!")
    print("All requirements have been installed successfully.")
    
    if not flash_attn_success:
        print("\n📝 Note: flash-attn was not installed. This is okay for CPU-only usage.")

if __name__ == "__main__":
    main()
