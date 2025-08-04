# VLLM Qwen2.5-Omni Video Captioning - Complete Installation Guide

## 📋 Prerequisites

- **NVIDIA GPU** with CUDA support (8GB+ VRAM recommended for AWQ model)
- **CUDA 11.8+** or **CUDA 12.x**
- **Python 3.8+**
- **Conda** or **Miniconda** installed
- **Git** installed

## 🚀 Step-by-Step Installation

### Step 1: Create Conda Environment

```bash
# Create new conda environment (Python 3.9-3.12 supported)
conda create -n video-ds-qwen python=3.11
conda activate video-ds-qwen
```

### Step 2: Install CUDA Toolkit (if needed)

```bash
# Check if CUDA is available
nvcc --version

# If CUDA not available, install via conda
conda install -c nvidia cuda-toolkit=12.1
```

### Step 3: Install PyTorch with CUDA

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Step 4: Install Base Dependencies

```bash
# Install basic requirements
pip install transformers>=4.52.3
pip install accelerate
pip install toml
pip install setuptools_scm
```

### Step 5: Install VLLM (Choose Option A or B)

#### Option A: Standard VLLM (Simpler, Text Output Only)

```bash
pip install vllm>=0.8.5.post1
pip install qwen-omni-utils[decord]
```

#### Option B: Full Qwen2.5-Omni Support (Recommended)

```bash
# Clone VLLM fork
git clone -b qwen2_omni_public https://github.com/fyabc/vllm.git
cd vllm
git checkout de8f43fbe9428b14d31ac5ec45d065cd3e5c3ee0

# Install additional dependencies
pip install torchdiffeq resampy x_transformers qwen-omni-utils[decord]

# Install VLLM requirements
pip install -r requirements/cuda.txt
pip install --upgrade setuptools wheel

# Install VLLM
pip install .

# Go back to project directory
cd ..
```

### Step 6: Download Project Files

```bash
# Create project directory
mkdir video-captioning
cd video-captioning

# Create required directories
mkdir videos captions models

# Download/create project files (config.toml, main.py, etc.)
# Copy the files from our previous artifacts
```

### Step 7: Configure the System

```bash
# Set environment variable for VLLM V0 engine
export VLLM_USE_V1=0

# Add to ~/.bashrc for permanent setting
echo 'export VLLM_USE_V1=0' >> ~/.bashrc
```

### Step 8: Download Model (Optional - Auto-downloads on first run)

```bash
# Pre-download model to avoid first-run delay
python -c "
from transformers import AutoTokenizer, AutoProcessor
model_name = 'Qwen/Qwen2.5-Omni-7B-AWQ'
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
print('Model downloaded successfully!')
"
```

## 🔧 Configuration

### Edit config.toml

```toml
[model]
name = "Qwen/Qwen2.5-Omni-7B-AWQ"
model_path = "./models/Qwen2.5-Omni-7B-AWQ"
trust_remote_code = true
dtype = "auto"
max_model_len = 8192

[hardware]
gpu_memory_utilization = 0.9
tensor_parallel_size = 1

[generation]
temperature = 0.7
max_tokens = 512
top_p = 0.9

[paths]
input_dir = "./videos"              # Change to your video folder
output_dir = "./captions"           # Change to your output folder
video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

[processing]
batch_size = 1
use_audio_in_video = true

[prompts]
system_prompt = "You are a helpful assistant that describes videos in detail."
user_prompt = "Please describe this video in detail, including what you see and hear."
```

## 🧪 Test Installation

### Quick Test

```bash
# Test VLLM installation
python -c "
import vllm
from vllm import LLM
print('VLLM imported successfully!')
"

# Test Qwen utilities
python -c "
import qwen_omni_utils
print('Qwen utilities imported successfully!')
"
```

### Full Test with Sample Video

```bash
# Place a test video in ./videos/ folder
# Run the captioning system
python main.py
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce GPU memory utilization in config.toml
gpu_memory_utilization = 0.7  # or lower
```

**Import Errors:**
```bash
# Reinstall with clean environment
conda deactivate
conda env remove -n video-captioning
# Start over from Step 1
```

**VLLM V1 Engine Issues:**
```bash
# Ensure V0 engine is used
export VLLM_USE_V1=0
```

**Transformers Version Conflicts:**
```bash
# Force install specific version
pip install transformers==4.52.3 --force-reinstall
```

## 📊 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU Memory | 6GB | 12GB+ |
| System RAM | 16GB | 32GB+ |
| Disk Space | 20GB | 50GB+ |
| CUDA | 11.8+ | 12.1+ |

## 🎯 Final Directory Structure

```
video-captioning/
├── config.toml
├── main.py
├── video_loader.py
├── vllm_inference.py
├── output_writer.py
├── videos/                 # Put your videos here
├── captions/               # Generated captions appear here
└── models/                 # Model cache
```

## ✅ Ready to Run!

```bash
# Activate environment
conda activate video-captioning

# Run the video captioning system
python main.py
```

The system will automatically process all videos in the `videos` folder and save captions to the `captions` folder.