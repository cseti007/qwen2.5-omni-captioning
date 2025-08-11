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

from video_loader import get_video_files, prepare_video_for_vllm
from image_loader import get_image_files, prepare_image_for_vllm
from vllm_inference import create_inference_engine
from output_writer import save_caption, check_existing_caption


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='🎯 %(asctime)s - %(levelname)s - %(message)s',  # Feltűnőbb format
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
    """
    Process media files in batches for better performance
    
    Args:
        media_files: List of media file paths
        inference_engine: VLLM inference engine
        config: Configuration dictionary
    
    Returns:
        Tuple of (successful_count, failed_count)
    """
    processing_config = config['processing']
    mode = processing_config.get('mode', 'video')
    batch_size = processing_config.get('batch_size', 4)
    
    # Get effective batch size for media type
    effective_batch_size = inference_engine.get_effective_batch_size(mode)
    
    successful = 0
    failed = 0
    
    # Process in chunks
    for i in range(0, len(media_files), effective_batch_size):
        batch = media_files[i:i + effective_batch_size]
        batch_paths = [str(f) for f in batch]
        
        logging.info(f"Processing batch {i//effective_batch_size + 1} ({len(batch)} files)")
        
        try:
            # Check which files need processing (skip existing captions)
            files_to_process = []
            files_to_skip = []
            
            for media_path in batch_paths:
                if check_existing_caption(media_path, config):
                    files_to_skip.append(Path(media_path).name)
                else:
                    files_to_process.append(media_path)
            
            if files_to_skip:
                logging.info(f"Skipping {len(files_to_skip)} files with existing captions")
                successful += len(files_to_skip)
            
            if not files_to_process:
                continue
            
            # Prepare media files
            prepared_paths = []
            for media_path in files_to_process:
                if mode == 'video':
                    prepared_path = prepare_video_for_vllm(media_path)
                elif mode == 'image':
                    prepared_path = prepare_image_for_vllm(media_path)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                prepared_paths.append(prepared_path)
            
            # Generate batch captions
            captions = inference_engine.generate_batch_captions(prepared_paths, mode)
            
            # Save captions
            for j, caption in enumerate(captions):
                if caption:  # Only save non-empty captions
                    original_path = files_to_process[j]
                    output_path = save_caption(caption, original_path, config)
                    logging.info(f"Batch saved: {Path(original_path).name} -> {Path(output_path).name}")
                    successful += 1
                else:
                    logging.warning(f"Empty caption for: {Path(files_to_process[j]).name}")
                    failed += 1
                    
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            # Fallback to single processing for this batch
            logging.info("Falling back to single file processing for this batch")
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
    """
    Process a single media file (video or image)
    
    Args:
        media_path: Path to media file
        inference_engine: VLLM inference engine
        config: Configuration dictionary
    
    Returns:
        True if successful, False otherwise
    """
    media_file = Path(media_path)
    
    try:
        # Check if caption already exists
        if check_existing_caption(media_path, config):
            logging.info(f"Caption already exists, skipping: {media_file.name}")
            return True
        
        # Determine media type and prepare accordingly
        processing_config = config['processing']
        mode = processing_config.get('mode', 'video')
        
        if mode == 'video':
            prepared_media_path = prepare_video_for_vllm(media_path)
            media_type = 'video'
        elif mode == 'image':
            prepared_media_path = prepare_image_for_vllm(media_path)
            media_type = 'image'
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
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
        
        if config['processing'].get('skip_errors', True):
            logging.info("Continuing with next file...")
            return False
        else:
            raise


def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging()
        
        # Get processing mode
        processing_config = config['processing']
        mode = processing_config.get('mode', 'video')
        
        logging.info("Starting VLLM Qwen2.5-Omni Media Captioning")
        logging.info(f"Processing mode: {mode.upper()}")
        logging.info(f"Config file: {args.config}")
        logging.info(f"Model: {config['model']['name']}")
        logging.info(f"Input dir: {config['paths']['input_dir']}")
        logging.info(f"Output dir: {config['paths']['output_dir']}")
        
        # Log conversation settings
        conversation_config = config.get('conversation', {})
        enable_multi_round = conversation_config.get('enable_multi_round', False)
        rounds = conversation_config.get('rounds', 1)
        
        if enable_multi_round:
            logging.info(f"Multi-round conversation: ENABLED ({rounds} rounds)")
        else:
            logging.info("Multi-round conversation: DISABLED (single round)")

        # Get media files based on mode
        if mode == 'video':
            media_files = get_video_files(
                config['paths']['input_dir'],
                config['processing']['video_extensions']
            )
            media_type_name = "video"
        elif mode == 'image':
            media_files = get_image_files(
                config['paths']['input_dir'],
                config['processing']['image_extensions']
            )
            media_type_name = "image"
        else:
            raise ValueError(f"Unsupported processing mode: {mode}")
        
        if not media_files:
            logging.warning(f"No {media_type_name} files found in input directory")
            return
        
        logging.info(f"Found {len(media_files)} {media_type_name} files")
        
        # Initialize inference engine
        logging.info("Initializing VLLM inference engine...")
        inference_engine = create_inference_engine(config)
        
        # Check if batch processing is enabled
        batch_mode = processing_config.get('batch_mode', False)
        batch_size = processing_config.get('batch_size', 1)
        
        if batch_mode and len(media_files) > 1 and batch_size > 1:
            logging.info(f"Batch processing enabled (batch_size: {batch_size})")
            # Process media files in batches
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
                logging.info(f"Processing {i}/{len(media_files)}: {media_file.name}")
                
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