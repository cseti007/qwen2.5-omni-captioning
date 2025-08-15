# VLLM Qwen2.5-Omni Captioning

**Note:** This is a fun project built with vibecoding. I'm not a developer, so I might not be able to help with every technical issue.

This is a node-based image and video captioning tool using Qwen2.5-Omni models with VLLM backend.

## ToC
- [VLLM Qwen2.5-Omni Captioning](#vllm-qwen25-omni-captioning)
  - [ToC](#toc)
  - [Features](#features)
  - [Model support](#model-support)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Command Line](#command-line)
    - [Web Interface](#web-interface)
  - [Configuration](#configuration)
    - [System Config (`config.toml`)](#system-config-configtoml)
    - [Prompts Config (`example_prompts/prompts.toml`)](#prompts-config-example_promptspromptstoml)

## Features

- **Multimodal**: Video and image captioning
- **Web GUI**: Gradio interface with gallery view
- **Batch processing**: Multiple files simultaneously
- **Multi-round conversation**: The system processes media through multiple sequential rounds (configurable), where each round refines and improves the caption based on the previous round's output.
- **Multiple outputs**: TXT, CSV, JSON formats

## Model support
- Qwen2.5-Omni-7B-AWQ - It was only tested with this  model.
However, the code should work with the other models from the Qwen2.5-Omni family out of the box.

## Installation

### Prerequisites
- Python 3.11
- CUDA-compatible GPU

### Setup
```bash
git clone <repository-url>
cd <repository-name>
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Usage

### Command Line
```bash
python main.py
python main.py --config custom_config.toml
```

### Web Interface
```bash
python gradio_app.py
```
Access at: http://localhost:7860

**Note:** The GUI provides basic functionality (folder selection, prompt template switching, output formats). For advanced configuration and custom prompts, edit the config files directly.

**Note**: The model loads during the first captioning task, so the initial process takes longer. Check the console output to monitor the loading progress.

## Configuration

### System Config (`config.toml`)
```toml
[model]
name = "Qwen/Qwen2.5-Omni-7B-AWQ"           # Model name from HuggingFace
# model_path = "./models/Qwen/Qwen2.5-Omni-7B-AWQ"  # Local model path (optional)
trust_remote_code = true                     # Required for Qwen models
dtype = "auto"                               # Automatic precision selection
max_model_len = 32768                        # Maximum context length

[hardware]
gpu_memory_utilization = 0.8                # GPU memory usage (0.6-0.9)
tensor_parallel_size = 1                    # Multi-GPU parallel processing
vllm_engine = "v1"                          # "v0" | "v1" - VLLM engine version

[generation]
temperature = 0.0                           # Randomness in generation (0.0 = deterministic)
max_tokens = 512                            # Maximum tokens per response
top_p = 0.85                                # Nucleus sampling parameter

[paths]
input_dir = "./videos"                      # Directory containing media files
output_dir = "./captions/videos"            # Output directory for captions
output_formats = ["txt", "csv", "json"]     # Multiple output formats
# output_format = "csv"                     # Single format option (alternative)

[processing]
mode = "video"                              # "video" | "image" | "mixed"
use_audio_in_video = false                  # Enable audio processing in videos
overwrite_existing = true                   # Overwrite existing caption files
fps = 16.0                                  # Video frame rate for processing
batch_size = 2                             # Files processed simultaneously
batch_mode = true                           # Enable batch processing for speed
save_conversations = true                   # Save detailed conversation logs
max_num_batched_tokens = 16384             # V1 optimization: token budget per step

# Supported file extensions
video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

[conversation]
enable_multi_round = true                   # Enable multi-round conversation

[files]
prompts_config = "./example_prompts/prompts.toml"  # Path to prompts configuration
```

### Prompts Config (`example_prompts/prompts.toml`)
```toml
[general]
trigger_word = "MyTr1gg3r"                  # Optional trigger word for captions

[round1]
mode = "multimodal"                         # "multimodal" | "text" - input type
system_prompt = "You are a highly attentive assistant that describes images and videos with extreme frame-by-frame precision."
user_prompt = '''
Describe the video in extreme detail, with these guidelines:

**Object and Character Details:**
    * Don't refer to characters as 'individual', 'characters' and 'persons', instead always use their gender or refer to them with their gender.
    * Describe the appearance in detail
    * What notable objects are present?

**Actions and Movement:**
    * Describe ALL movements, no matter how subtle.
    * Specify the exact type of movement (walking, running, etc.).
    * Note the direction and speed of movements.

**Background Elements:**
    * Describe the setting and environment.
    * Note any environmental changes.

**Visual Style:**
    * Describe the lighting and color palette.
    * Note any special effects or visual treatments.
    * What is the overall style of the video? (e.g., realistic, animated, artistic, documentary)

**Camera Work:**
    * Describe EVERY camera angle change.
    * Note the distance from subjects (close-up, medium, wide shot).
    * Describe any camera movements (pan, tilt, zoom).

**Scene Transitions:**
    * How does each shot transition to the next?
    * Note any changes in perspective or viewing angle.

Please be extremely specific and detailed in your description. If you notice any movement or changes, describe them explicitly.
'''

[round2]
mode = "text"                               # Text-only mode uses previous round's output
system_prompt = "You are an expert video caption refiner. Your task is to improve video captions based on specific instructions."
user_prompt = '''
    **Summarize the generated caption in 500 tokens, but with the following modifications:**
        * Write continuously, don't use multiple paragraphs, make the text form one coherent whole
        * Call the main character: {trigger_word}
        * Do not mention your task
        * Don't use references to video such as "the video begins" or "the video features" etc., but keep those sentences meaningful
        * Mention the clothing details of the characters
        * Use only declarative sentences
        * Don't mention the style of the video
        * Don't mention the name of the characters
'''
