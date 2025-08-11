# VLLM Qwen2.5-Omni Captioning

This is a node based captioning tool using the multimodal Qwen2.5-Omni 7B model.

- [VLLM Qwen2.5-Omni Captioning](#vllm-qwen25-omni-captioning)
  - [Features](#features)
    - [Multimodal support](#multimodal-support)
    - [Advanced features](#advanced-features)
  - [Installation \& Setup](#installation--setup)
    - [Prerequisites](#prerequisites)
      - [Python 3.11 Installation](#python-311-installation)
      - [Clone this repo](#clone-this-repo)
    - [Virtual Environment Setup](#virtual-environment-setup)
      - [1. Create Virtual Environment](#1-create-virtual-environment)
      - [2. Activate Virtual Environment](#2-activate-virtual-environment)
    - [Dependencies Installation](#dependencies-installation)
    - [GPU Setup Verification](#gpu-setup-verification)
      - [Check CUDA availability](#check-cuda-availability)
  - [Quick Start](#quick-start)
    - [Basic Usage](#basic-usage)
  - [Configuration](#configuration)
    - [System Configuration (`config.toml`)](#system-configuration-configtoml)
    - [Prompts Configuration (`prompts.toml`)](#prompts-configuration-promptstoml)
  - [Performance Optimization](#performance-optimization)
    - [Batch Processing](#batch-processing)
    - [GPU Settings](#gpu-settings)
    - [Processing Modes](#processing-modes)
  - [Advanced Usage](#advanced-usage)
    - [Multi-Round Conversation](#multi-round-conversation)
    - [Custom Prompts](#custom-prompts)
    - [Output Formats](#output-formats)
      - [TXT Format](#txt-format)
      - [CSV Format](#csv-format)
      - [JSON Format](#json-format)
  - [Acknowledgments](#acknowledgments)

## Features
### Multimodal support
- Video captioning
- Image captioning

### Advanced features
- Batch processing: process multiple files simultaneously
- VLLM backend
- Multi-round conversation: kifejteni bővebben
- Multiple output formats: can provide TXT, CSV, JSON output simultaneously

## Installation & Setup

### Prerequisites

#### Python 3.11 Installation

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**CentOS/RHEL/Fedora:**
```bash
# Fedora
sudo dnf install python3.11 python3.11-venv

# CentOS/RHEL (requires EPEL)
sudo yum install epel-release
sudo yum install python311 python311-venv
```

**macOS:**
```bash
# Using Homebrew
brew install python@3.11

# Using pyenv
pyenv install 3.11.0
pyenv global 3.11.0
```

**Windows:**
- Download Python 3.11 from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"
- Verify with: `python --version`

#### Clone this repo

```bash
git clone <repository-url>
cd <repository-name>
```

### Virtual Environment Setup

#### 1. Create Virtual Environment
```bash
# Navigate to project directory
cd <repository-name>

# Create virtual environment
python3.11 -m venv venv
```

#### 2. Activate Virtual Environment

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
# Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt when activated.

### Dependencies Installation

```bash
pip install -r requirements.txt
```

### GPU Setup Verification

#### Check CUDA availability
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```
## Quick Start

### Basic Usage
```bash
# Process images
python main.py

# Process videos  
python main.py --config config_video.toml

# Custom configuration
python main.py --config my_custom_config.toml
```

## Configuration

### System Configuration (`config.toml`)
```toml
[model]
name = "Qwen/Qwen2.5-Omni-7B-AWQ"
trust_remote_code = true
dtype = "auto"
max_model_len = 24576

[hardware]
gpu_memory_utilization = 0.9
tensor_parallel_size = 1

[paths]
input_dir = "./images"
output_dir = "./captions"
output_formats = ["txt", "csv", "json"]

[processing]
mode = "image"  # "video" | "image"
batch_size = 4
batch_mode = true
overwrite_existing = true

[conversation]
enable_multi_round = true
rounds = 2

[files]
prompts_config = "prompts.toml"
```

### Prompts Configuration (`prompts.toml`)
Separate configuration file for prompts allows easy customization:
- **Video prompts**: Specialized for stop-motion and video analysis
- **Image prompts**: Optimized for art and photography analysis
- **Multiple prompt sets**: Alternative configurations

## Performance Optimization

### Batch Processing
- **Automatic scaling**: Videos use smaller batches due to memory requirements
- **Smart fallback**: Falls back to single processing if batch fails
- **Memory aware**: Adjusts batch size based on GPU memory

### GPU Settings
```toml
[hardware]
gpu_memory_utilization = 0.9  # Adjust based on your GPU
tensor_parallel_size = 1       # Increase for multi-GPU setups
```

### Processing Modes
```toml
[processing]
batch_size = 4
batch_mode = true     # Enable for performance boost
```

## Advanced Usage

### Multi-Round Conversation
The system supports multi-round caption refinement:
1. **Round 1**: Detailed analysis of visual content
2. **Round 2**: Refined, polished caption suitable for catalogs

### Custom Prompts
Create custom prompt configurations for different use cases:
- Art analysis prompts
- Technical documentation
- Social media captions
- Academic descriptions

### Output Formats

#### TXT Format
Plain text captions for simple use cases.

#### CSV Format  
Consolidated dataset with filename and caption columns.

#### JSON Format
Rich metadata including:
- Processing timestamp
- Model configuration
- Technical details
- Original file paths

## Acknowledgments

- **Qwen Team**: For the excellent Qwen2.5-Omni model
- **VLLM Team**: For the high-performance inference engine