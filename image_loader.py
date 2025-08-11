"""
Simple Image Loader for VLLM Qwen2.5-Omni Image Captioning
Uses built-in VLLM image handling - no manual preprocessing needed
"""

from pathlib import Path
from typing import List
import logging


def get_image_files(input_dir: str, extensions: List[str]) -> List[Path]:
    """Get all image files from input directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def prepare_image_for_vllm(image_path: str) -> str:
    """
    Prepare image path for VLLM inference
    VLLM handles image preprocessing automatically, so we just return the path
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    logging.info(f"Preparing image: {image_file.name}")
    
    # VLLM automatically handles:
    # - Image loading
    # - Format conversion to numpy arrays
    # - Preprocessing for the model
    
    # Return path as string without file:// prefix
    return str(image_file.resolve())