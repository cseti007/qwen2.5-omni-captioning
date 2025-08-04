"""
Main Pipeline for VLLM Qwen2.5-Omni Video Captioning
Simple execution flow: load config -> process videos -> generate captions -> save results
"""

import sys
import logging
import toml
from pathlib import Path
from typing import Dict, Any

from video_loader import get_video_files, prepare_video_for_vllm
from vllm_inference import create_inference_engine
from output_writer import save_caption, check_existing_caption


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load configuration from TOML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    # Ensure output directory exists
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    return config


def process_single_video(video_path: str, inference_engine, config: Dict[str, Any]) -> bool:
    """
    Process a single video file
    
    Args:
        video_path: Path to video file
        inference_engine: VLLM inference engine
        config: Configuration dictionary
    
    Returns:
        True if successful, False otherwise
    """
    video_file = Path(video_path)
    
    try:
        # Check if caption already exists
        if check_existing_caption(video_path, config):
            logging.info(f"Caption already exists, skipping: {video_file.name}")
            return True
        
        # Prepare video for VLLM
        prepared_video_path = prepare_video_for_vllm(video_path)
        
        # Generate caption
        caption = inference_engine.generate_caption(prepared_video_path)
        
        if not caption:
            logging.warning(f"Empty caption generated for: {video_file.name}")
            return False
        
        # Save caption
        output_path = save_caption(caption, video_path, config)
        logging.info(f"Successfully processed: {video_file.name} -> {Path(output_path).name}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to process {video_file.name}: {e}")
        
        if config['processing'].get('skip_errors', True):
            logging.info("Continuing with next video...")
            return False
        else:
            raise


def main():
    """Main execution function"""
    try:
        # Load configuration
        config = load_config()
        
        # Setup logging
        setup_logging()
        
        logging.info("Starting VLLM Qwen2.5-Omni Video Captioning")
        logging.info(f"Model: {config['model']['name']}")
        logging.info(f"Input dir: {config['paths']['input_dir']}")
        logging.info(f"Output dir: {config['paths']['output_dir']}")
        
        # Get video files
        video_files = get_video_files(
            config['paths']['input_dir'],
            config['paths']['video_extensions']
        )
        
        if not video_files:
            logging.warning("No video files found in input directory")
            return
        
        logging.info(f"Found {len(video_files)} video files")
        
        # Initialize inference engine
        logging.info("Initializing VLLM inference engine...")
        inference_engine = create_inference_engine(config)
        
        # Process videos
        successful = 0
        failed = 0
        
        for i, video_file in enumerate(video_files, 1):
            logging.info(f"Processing {i}/{len(video_files)}: {video_file.name}")
            
            if process_single_video(str(video_file), inference_engine, config):
                successful += 1
            else:
                failed += 1
        
        # Cleanup
        inference_engine.cleanup()
        
        # Summary
        logging.info("=" * 50)
        logging.info("Processing completed!")
        logging.info(f"Successful: {successful}")
        logging.info(f"Failed: {failed}")
        logging.info(f"Total: {len(video_files)}")
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()