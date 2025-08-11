"""
Output Writer Module for VLLM Qwen2.5-Omni Video Captioning
Handles saving captions to various formats
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def save_caption(caption: str, video_path: str, config: Dict[str, Any]) -> str:
    """
    Save caption to output file(s) - supports multiple formats simultaneously
    
    Args:
        caption: Generated caption text
        video_path: Original video file path
        config: Configuration dictionary
    
    Returns:
        Path to saved caption file (first format if multiple)
    """
    # Add trigger word to caption if configured
    trigger_word = config.get('prompts', {}).get('trigger_word', '')
    if trigger_word:
        caption = f"{trigger_word}. {caption}"
    
    video_file = Path(video_path)
    paths_config = config['paths']
    
    # Create output directory
    output_dir = Path(paths_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle both single format (backward compatibility) and multiple formats
    output_formats = paths_config.get('output_formats', None)
    if output_formats is None:
        # Fallback to single format for backward compatibility
        output_formats = [paths_config.get('output_format', 'txt')]
    elif isinstance(output_formats, str):
        # Handle case where it's a single string instead of list
        output_formats = [output_formats]
    
    saved_paths = []
    
    # Process each output format
    for output_format in output_formats:
        if output_format == 'csv':
            # For CSV, use a single file for all captions
            output_filename = "captions.csv"
            output_path = output_dir / output_filename
            _save_as_csv(caption, video_file.name, output_path)
        else:
            # For other formats, use individual files
            output_filename = f"{video_file.stem}.{output_format}"
            output_path = output_dir / output_filename
            
            if output_format == 'txt':
                _save_as_text(caption, output_path)
            elif output_format == 'json':
                _save_as_json(caption, video_path, output_path, config)
            else:
                # Default to text for unknown formats
                _save_as_text(caption, output_path)
        
        saved_paths.append(str(output_path))
        logging.info(f"Caption saved ({output_format}): {output_path}")
    
    # Return first saved path for compatibility
    return saved_paths[0] if saved_paths else ""


def _save_as_text(caption: str, output_path: Path):
    """Save caption as plain text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(caption)


def _save_as_json(caption: str, video_path: str, output_path: Path, config: Dict[str, Any]):
    """Save caption as JSON file with metadata"""
    video_file = Path(video_path)
    
    # Create JSON structure
    caption_data = {
        "video_file": video_file.name,
        "video_path": str(video_file.absolute()),
        "caption": caption,
        "timestamp": datetime.now().isoformat(),
        "model": config['model']['name'],
        "config": {
            "temperature": config['generation']['temperature'],
            "max_tokens": config['generation']['max_tokens'],
            "use_audio_in_video": config['processing'].get('use_audio_in_video', False)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(caption_data, f, indent=2, ensure_ascii=False)


def _save_as_csv(caption: str, filename: str, output_path: Path):
    """Save caption as CSV file with filename and caption columns"""
    # Check if file exists to determine if we need to write header
    write_header = not output_path.exists()
    
    # Open in append mode to continuously add new captions
    with open(output_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header only if file is new
        if write_header:
            writer.writerow(['filename', 'caption'])
        
        # Write the caption row
        writer.writerow([filename, caption])


def check_existing_caption(video_path: str, config: Dict[str, Any]) -> bool:
    """
    Check if caption already exists for the video
    
    Args:
        video_path: Path to video file
        config: Configuration dictionary
    
    Returns:
        True if caption exists and overwrite is disabled
    """
    if config['processing'].get('overwrite_existing', True):
        return False
    
    video_file = Path(video_path)
    paths_config = config['paths']
    
    # Handle both single format and multiple formats
    output_formats = paths_config.get('output_formats', None)
    if output_formats is None:
        output_formats = [paths_config.get('output_format', 'txt')]
    elif isinstance(output_formats, str):
        output_formats = [output_formats]
    
    # Check if ANY of the formats already exist
    for output_format in output_formats:
        if output_format == 'csv':
            # For CSV, check if the filename already exists in the CSV
            output_dir = Path(paths_config['output_dir'])
            csv_path = output_dir / "captions.csv"
            
            if csv_path.exists():
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row['filename'] == video_file.name:
                                return True
                except Exception:
                    continue
        else:
            # For other formats, check individual files
            output_dir = Path(paths_config['output_dir'])
            output_filename = f"{video_file.stem}.{output_format}"
            output_path = output_dir / output_filename
            
            if output_path.exists():
                return True
    
    return False