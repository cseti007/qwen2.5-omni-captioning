#!/usr/bin/env python3
"""
Install script for qwen2.5-omni-captioning requirements.
This script ensures proper installation order to avoid flash-attn build issues.
"""

import subprocess
import sys
import os

def run_pip_install(packages, description="", verbose=True):
    """Run pip install with error handling and progress monitoring"""
    if isinstance(packages, str):
        packages = [packages]
    
    print(f"\n{'='*50}")
    print(f"Installing {description}...")
    print(f"Packages: {', '.join(packages)}")
    print(f"{'='*50}")
    
    try:
        # Add verbose flag and progress indicator
        cmd = [sys.executable, "-m", "pip", "install"]
        if verbose:
            cmd.extend(["-v", "--progress-bar", "on"])
        cmd.extend(packages)
        
        print("Command:", " ".join(cmd))
        print("⏳ This may take several minutes, especially for PyTorch with CUDA support...")
        print("💡 If it seems stuck, it's likely downloading large packages in the background.")
        
        # Use Popen for real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        if process.stdout:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
        
        return_code = process.poll()
        if return_code == 0:
            print(f"✅ Successfully installed {description}")
            return True
        else:
            print(f"❌ Failed to install {description} (exit code: {return_code})")
            return False
            
    except Exception as e:
        print(f"❌ Failed to install {description}: {e}")
        return False

def main():
    print("🚀 Starting qwen2.5-omni-captioning requirements installation...")
    print("⚠️  Installing VLLM first to avoid dependency conflicts...")
    
    # Step 1: Install VLLM FIRST (it will install its preferred PyTorch/CUDA versions)
    print("\n🎯 Step 1: Installing VLLM with its dependencies")
    print("This will automatically install the correct PyTorch and CUDA versions for VLLM")
    vllm_deps = [
        "wheel",  # Required for building
        "vllm>=0.8.5"
    ]
    
    if not run_pip_install(vllm_deps, "VLLM (will install compatible PyTorch automatically)"):
        sys.exit(1)
    
    # Step 2: Install packages that must be compatible with VLLM's PyTorch
    print("\n🎯 Step 2: Installing packages compatible with VLLM's PyTorch version")
    torch_compatible = [
        "transformers>=4.52.3",
        "accelerate",
        "flash-attn>=2.0.0",  # Move flash-attn here, after torch is properly installed
    ]
    
    # Install without flash-attn first, then try flash-attn separately
    basic_torch_deps = torch_compatible[:-1]  # Without flash-attn
    if not run_pip_install(basic_torch_deps, "PyTorch-compatible packages"):
        sys.exit(1)
    
    # Try flash-attn separately (it's often problematic)
    print("\n🎯 Step 2b: Installing flash-attn (may fail on CPU-only systems)")
    flash_success = run_pip_install("flash-attn>=2.0.0", "flash-attn", verbose=False)
    
    # Step 3: Install Qwen utilities
    print("\n🎯 Step 3: Installing Qwen utilities")
    if not run_pip_install("qwen-omni-utils[decord]", "Qwen utilities with video support"):
        sys.exit(1)
    
    # Step 4: Install remaining dependencies (should be safe now)
    print("\n🎯 Step 4: Installing remaining dependencies")
    other_deps = [
        "toml",
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
    
    if not run_pip_install(other_deps, "remaining dependencies"):
        sys.exit(1)
    
    print("\n🎉 Installation completed!")
    print("All requirements have been installed successfully.")
    
    if not flash_success:
        print("\n📝 Note: flash-attn was not installed. This is okay for CPU-only usage.")

if __name__ == "__main__":
    main()
