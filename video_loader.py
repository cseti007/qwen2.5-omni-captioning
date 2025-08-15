"""
Simple Video Loader for VLLM Qwen2.5-Omni Video Captioning
Uses built-in VLLM video handling - no manual preprocessing needed
"""

from pathlib import Path
from typing import List
import logging


def get_video_files(input_dir: str, extensions: List[str]) -> List[Path]:
    """Get all video files from input directory"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    video_files = []
    for ext in extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    return sorted(video_files)


def prepare_video_for_vllm(video_path: str) -> str:
    """
    Prepare video path for VLLM inference
    VLLM handles video preprocessing automatically, so we just return the path
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    logging.info(f"Preparing video: {video_file.name}")
    
    # VLLM automatically handles:
    # - Video loading
    # - Frame extraction
    # - Audio extraction (if use_audio_in_video=True)
    # - Format conversion to numpy arrays
    # - Preprocessing for the model
    
    # Return path as string without file:// prefix
    return str(video_file.resolve())