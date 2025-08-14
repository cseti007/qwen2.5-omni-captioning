"""
Main Pipeline for VLLM Qwen2.5-Omni Video/Image Captioning
Simple execution flow: load config -> process media -> generate captions -> save results
"""

import sys
import logging
import toml
import argparse
from pathlib import Path
from typing import Dict, Any, List

from video_loader import prepare_video_for_vllm
from image_loader import prepare_image_for_vllm
from vllm_inference import create_inference_engine, list_all_supported_files, detect_media_type
from output_writer import save_caption, check_existing_caption


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='🎯 %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def load_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load configuration from TOML files (system + prompts)"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load system configuration
    with open(config_file, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    # Load prompts configuration
    prompts_file = config.get('files', {}).get('prompts_config', 'prompts.toml')
    prompts_path = Path(prompts_file)
    
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts_config = toml.load(f)
    
    # Merge prompts into main config
    config['prompts'] = prompts_config
    
    # Ensure output directory exists
    Path(config['paths']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Loaded system config: {config_path}")
    logging.info(f"Loaded prompts config: {prompts_file}")
    
    return config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="VLLM Qwen2.5-Omni Video/Image Captioning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Use default config.toml
  python main.py --config my_config.toml  # Use custom config file
  python main.py -c custom.toml           # Short form
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.toml',
        help='Path to configuration file (default: config.toml)'
    )
    
    return parser.parse_args()


def process_batch_media(media_files: List[Path], inference_engine, config: Dict[str, Any]) -> tuple[int, int]:
    """Process media files in batches with auto-detection"""
    batch_size = config['processing'].get('batch_size', 4)
    effective_batch_size = inference_engine.get_effective_batch_size("mixed")
    
    successful = 0
    failed = 0
    
    # Process in chunks
    for i in range(0, len(media_files), effective_batch_size):
        batch = media_files[i:i + effective_batch_size]
        batch_paths = [str(f) for f in batch]
        
        logging.info(f"Processing batch {i//effective_batch_size + 1} ({len(batch)} files)")
        
        try:
            # Check existing captions and group by media type
            video_files = []
            image_files = []
            files_to_skip = []
            
            for media_path in batch_paths:
                if check_existing_caption(media_path, config):
                    files_to_skip.append(Path(media_path).name)
                    continue
                
                media_type = detect_media_type(media_path)
                if media_type == "video":
                    video_files.append(media_path)
                else:
                    image_files.append(media_path)
            
            if files_to_skip:
                logging.info(f"Skipping {len(files_to_skip)} files with existing captions")
                successful += len(files_to_skip)
            
            # Process video files in batch
            if video_files:
                prepared_videos = [prepare_video_for_vllm(path) for path in video_files]
                video_captions = inference_engine.generate_batch_captions(prepared_videos, "video")
                
                for j, caption in enumerate(video_captions):
                    if caption:
                        output_path = save_caption(caption, video_files[j], config)
                        logging.info(f"Batch saved: {Path(video_files[j]).name} -> {Path(output_path).name}")
                        successful += 1
                    else:
                        failed += 1
            
            # Process image files in batch
            if image_files:
                prepared_images = [prepare_image_for_vllm(path) for path in image_files]
                image_captions = inference_engine.generate_batch_captions(prepared_images, "image")
                
                for j, caption in enumerate(image_captions):
                    if caption:
                        output_path = save_caption(caption, image_files[j], config)
                        logging.info(f"Batch saved: {Path(image_files[j]).name} -> {Path(output_path).name}")
                        successful += 1
                    else:
                        failed += 1
                    
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            # Fallback to single processing
            for media_file in batch:
                try:
                    if process_single_media(str(media_file), inference_engine, config):
                        successful += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
    
    return successful, failed


def process_single_media(media_path: str, inference_engine, config: Dict[str, Any]) -> bool:
    """Process a single media file with auto-detection"""
    media_file = Path(media_path)
    
    try:
        # Check if caption already exists
        if check_existing_caption(media_path, config):
            logging.info(f"Caption already exists, skipping: {media_file.name}")
            return True
        
        # Auto-detect media type and prepare accordingly
        media_type = detect_media_type(media_path)
        
        if media_type == 'video':
            prepared_media_path = prepare_video_for_vllm(media_path)
        elif media_type == 'image':
            prepared_media_path = prepare_image_for_vllm(media_path)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
        
        # Generate caption
        caption = inference_engine.generate_caption(prepared_media_path, media_type)
        
        if not caption:
            logging.warning(f"Empty caption generated for: {media_file.name}")
            return False
        
        # Save caption
        output_path = save_caption(caption, media_path, config)
        logging.info(f"Successfully processed: {media_file.name} -> {Path(output_path).name}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to process {media_file.name}: {e}")
        return False


def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging()
        
        logging.info("Starting VLLM Qwen2.5-Omni Media Captioning")
        logging.info(f"Config file: {args.config}")
        logging.info(f"Model: {config['model']['name']}")
        logging.info(f"Input dir: {config['paths']['input_dir']}")
        logging.info(f"Output dir: {config['paths']['output_dir']}")
        
        # Log conversation settings
        conversation_config = config.get('conversation', {})
        enable_multi_round = conversation_config.get('enable_multi_round', False)
        
        if enable_multi_round:
            logging.info("Multi-round conversation: ENABLED (rounds auto-detected from prompts)")
        else:
            logging.info("Multi-round conversation: DISABLED (single round)")

        # Auto-detect all supported media files
        media_files = list_all_supported_files(config['paths']['input_dir'])
        
        if not media_files:
            logging.warning("No supported media files found in input directory")
            return
        
        # Count by type for logging
        video_count = sum(1 for f in media_files if detect_media_type(str(f)) == "video")
        image_count = len(media_files) - video_count
        
        logging.info(f"Found {len(media_files)} media files ({video_count} videos, {image_count} images)")
        
        # Initialize inference engine
        logging.info("Initializing VLLM inference engine...")
        inference_engine = create_inference_engine(config)
        
        # Check if batch processing is enabled
        processing_config = config['processing']
        batch_mode = processing_config.get('batch_mode', False)
        batch_size = processing_config.get('batch_size', 1)
        
        if batch_mode and len(media_files) > 1 and batch_size > 1:
            logging.info(f"Batch processing enabled (batch_size: {batch_size})")
            successful, failed = process_batch_media(media_files, inference_engine, config)
        else:
            if batch_mode:
                logging.info("Batch mode enabled but using single processing (insufficient files or batch_size=1)")
            else:
                logging.info("Single file processing mode")
            
            # Process media files individually
            successful = 0
            failed = 0
            
            for i, media_file in enumerate(media_files, 1):
                media_type = detect_media_type(str(media_file))
                logging.info(f"Processing {i}/{len(media_files)}: {media_file.name} ({media_type})")
                
                if process_single_media(str(media_file), inference_engine, config):
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
        logging.info(f"Total: {len(media_files)}")
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()