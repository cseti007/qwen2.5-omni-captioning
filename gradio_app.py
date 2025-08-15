"""
Gradio GUI for VLLM Qwen2.5-Omni Video/Image Captioning
Real-time Gallery interface with streaming results display and working stop function
"""

import os
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Dict, Any
import threading
import time
import logging
import toml

# Import existing modules
from main import load_config, create_inference_engine, process_single_media
from vllm_inference import list_all_supported_files, detect_media_type
from output_writer import check_existing_caption

# Set static paths for direct file serving
gr.set_static_paths(paths=["videos/", "images/", "test_media/", "captions/", "static/"])


# Global processing state for stop functionality
processing_state = {
    'is_processing': False,
    'stop_event': threading.Event(),
    'inference_engine': None
}


def create_status_msg(status, i, total, filename, media_type="", extra=""):
    """Create standardized status message"""
    type_info = f"Type: {media_type}\n" if media_type else ""
    return f"{status} ({i}/{total})\n\n{filename}\n\n{type_info}{extra}"


def convert_path_for_web(path):
    """Convert file path for web serving"""
    try:
        return str(path).replace('\\', '/') if Path(path).is_absolute() else str(Path(path).relative_to(Path.cwd())).replace('\\', '/')
    except ValueError:
        return str(path).replace('\\', '/')


def create_media_item(path, caption):
    """Create HTML for single media item with clean architecture"""
    web_path = convert_path_for_web(path)
    is_video = Path(path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # Clean structure - no positioning logic in HTML
    if is_video:
        return f'''
        <div class="gallery-item video">
            <div class="media-container">
                <video muted loop preload="auto" 
                       onmouseover="this.play().catch(e=>console.log('Video play failed:', e))" 
                       onmouseout="this.pause(); this.currentTime=0;"
                       onclick="if(this.paused) this.play(); else this.pause();">
                    <source src="/gradio_api/file={web_path}" type="video/mp4">
                </video>
                <div class="play-overlay">▶</div>
            </div>
            <div class="caption">{caption}</div>
        </div>'''
    else:
        return f'''
        <div class="gallery-item image">
            <div class="media-container">
                <img src="/gradio_api/file={web_path}" alt="Generated image" loading="lazy" onclick="window.open(this.src, '_blank');">
            </div>
            <div class="caption">{caption}</div>
        </div>'''


def create_hf_style_gallery(gallery_results):
    """Create HuggingFace-style widget gallery with compact HTML generation"""
    if not gallery_results:
        return '<div class="gallery-empty">No results yet...</div>'
    
    items = []
    for path, caption in gallery_results:
        if path == "error":
            items.append(f'<div class="gallery-item error"><div class="caption">{caption}</div></div>')
        else:
            items.append(create_media_item(path, caption))
    
    return f'<div class="gallery-container"><div class="gallery" id="gallery">{"".join(items)}</div></div>'


def process_folder_streaming(folder_path: str, output_path: str, prompt_template: str, output_formats: List[str]):
    """Generator function for real-time Gallery updates with streaming results"""
    if not folder_path or not Path(folder_path).exists():
        yield [("error", "⚠️ Invalid folder path!")]
        return
    
    # Load and configure using existing main.py function
    config = load_config()
    config['paths']['input_dir'] = folder_path
    config['paths']['output_dir'] = output_path
    config['files']['prompts_config'] = f"./example_prompts/{prompt_template}"
    config['conversation']['enable_multi_round'] = True
    config['paths']['output_formats'] = output_formats
    
    logging.info(f"📋 Using prompt template: {prompt_template}")
    
    # Find media files using existing function
    media_files = list_all_supported_files(folder_path)
    if not media_files:
        yield [("error", "⚠️ No supported media files found in folder!")]
        return
    
    # Load model using existing function
    try:
        if processing_state['inference_engine'] is None:
            processing_state['inference_engine'] = create_inference_engine(config)
    except Exception as e:
        yield [("error", f"⚠️ Model loading failed: {str(e)}")]
        return
    
    gallery_results = []
    
    # Process files with stop checking
    for i, media_file in enumerate(media_files):
        if processing_state['stop_event'].is_set():
            gallery_results.append((str(media_file), create_status_msg("🛑 STOPPED", i+1, len(media_files), media_file.name, extra="Processing stopped by user")))
            yield gallery_results
            break
        
        try:
            media_type = detect_media_type(str(media_file))
            
            gallery_results.append((str(media_file), create_status_msg("🔄 PROCESSING", i+1, len(media_files), media_file.name, media_type, "Generating caption...")))
            yield gallery_results
            
            if processing_state['stop_event'].is_set():
                gallery_results[-1] = (str(media_file), create_status_msg("🛑 STOPPED", i+1, len(media_files), media_file.name, extra="Stopped before processing"))
                yield gallery_results
                break
            
            success = process_single_media(str(media_file), processing_state['inference_engine'], config)
            
            if success:
                # Read caption from generated file
                media_stem = Path(media_file).stem
                caption_file = Path(config['paths']['output_dir']) / f"{media_stem}.txt"
                caption_text = caption_file.read_text(encoding='utf-8').strip() if caption_file.exists() else "Caption file not found"
                
                gallery_results[-1] = (str(media_file), create_status_msg("✅ COMPLETED", i+1, len(media_files), media_file.name, media_type, f"Caption:\n{caption_text}"))
            else:
                gallery_results[-1] = (str(media_file), create_status_msg("⚠️ FAILED", i+1, len(media_files), media_file.name, media_type, "Failed to generate caption"))
            
            yield gallery_results
                        
        except Exception as e:
            gallery_results[-1] = (str(media_file), create_status_msg("⚠️ ERROR", i+1, len(media_files), media_file.name, extra=f"Error: {str(e)}"))
            yield gallery_results
    
    successful = sum(1 for _, caption in gallery_results if "✅ COMPLETED" in caption)
    failed = len(gallery_results) - successful
    
    if gallery_results:
        first_item = gallery_results[0]
        summary_caption = f"{first_item[1]}\n\n📊 FINAL SUMMARY:\n✅ Successful: {successful}\n⚠️ Failed: {failed}"
        gallery_results[0] = (first_item[0], summary_caption)
    
    yield gallery_results


def list_prompt_templates():
    """List available prompt TOML files from example_prompts directory"""
    prompts_dir = Path("example_prompts")
    if not prompts_dir.exists():
        return ["prompts.toml"]
    
    toml_files = [f.name for f in prompts_dir.glob("*.toml")]
    return sorted(toml_files) if toml_files else ["prompts.toml"]


def validate_folder(folder_path: str):
    """Validate folder path and show file count using existing functions"""
    if not folder_path:
        return "📁 Please enter a folder path"
    
    path = Path(folder_path)
    if not path.exists():
        return "⚠️ Folder does not exist"
    
    if not path.is_dir():
        return "⚠️ Path is not a directory"
    
    try:
        media_files = list_all_supported_files(folder_path)
        video_count = sum(1 for f in media_files if detect_media_type(str(f)) == "video")
        image_count = len(media_files) - video_count
        
        if not media_files:
            return "⚠️ No supported media files found"
        
        return f"✅ Found {len(media_files)} files ({video_count} videos, {image_count} images)"
    except Exception as e:
        return f"⚠️ Error scanning folder: {str(e)}"


def process_folder_gui(folder_path: str, output_path: str, prompt_template: str, txt_format: bool, json_format: bool, csv_format: bool):
    """GUI wrapper for folder processing with real-time streaming"""
    # Determine output formats
    output_formats = []
    if txt_format:
        output_formats.append("txt")
    if json_format:
        output_formats.append("json") 
    if csv_format:
        output_formats.append("csv")
    
    if not output_formats:
        yield create_hf_style_gallery([("error", "⚠️ Please select at least one output format!")])
        return
    
    # Start processing
    processing_state['is_processing'] = True
    processing_state['stop_event'].clear()
    logging.info("Processing started")
    
    try:
        for gallery_results in process_folder_streaming(
            folder_path, output_path, prompt_template, output_formats
        ):
            yield create_hf_style_gallery(gallery_results)
    finally:
        processing_state['is_processing'] = False


def stop_processing_gui():
    """Stop button handler with logging"""
    print(f"\n{'='*50}")
    print(f"🛑 STOP BUTTON PRESSED - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    logging.info(f"Stop button clicked by user at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if processing_state['is_processing']:
        processing_state['stop_event'].set()
        print(f"ℹ️ Processing will stop after current file completes")
        logging.info("Stop signal sent - processing will halt")
    else:
        print(f"ℹ️ No active processing to stop")
        logging.info("Stop requested but no active processing found")
    
    print(f"📋 Stop request processed successfully")
    return "ℹ️ Stop requested - processing will complete current file and halt"


def create_interface():
    with gr.Blocks(
        title="VLLM Caption Generator", 
        css_paths=["static/gallery.css"],
        js="() => { if (!window.location.href.includes('__theme=dark')) { window.location.href += '?__theme=dark'; } }",
        theme=None
    ) as interface:
        
        gr.Markdown("# 🎯 VLLM Qwen2.5-Omni Caption Generator")
        gr.Markdown("Generate captions for images and videos using VLLM with real-time progress display")
        
        with gr.Row():
            # Left column - main inputs
            with gr.Column(scale=4):
                with gr.Group():
                    # Row 1: Input folder + status
                    with gr.Row():
                        default_input_path = "./videos"

                        folder_input = gr.Textbox(
                            label="📁 Input Folder Path",
                            placeholder="/path/to/your/media/folder",
                            value=default_input_path,
                            scale=3
                        )
                        
                        folder_status = gr.Textbox(
                            label="📊 Folder Status",
                            value=validate_folder(default_input_path),
                            interactive=False,
                            min_width=200,
                            scale=2
                        )
                    
                    # Row 2: Output folder + prompt template
                    with gr.Row():
                        output_input = gr.Textbox(
                            label="💾 Output Folder Path",
                            placeholder="/path/to/save/captions",
                            value="./captions",
                            scale=3
                        )
                        
                        prompt_template = gr.Dropdown(
                            label="📋 Prompt Template",
                            choices=list_prompt_templates(),
                            value="prompts.toml",
                            scale=2
                        )
            
            # Right column - output formats
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Output Formats")
                txt_format = gr.Checkbox(label="TXT files", value=True)
                json_format = gr.Checkbox(label="JSON files", value=False)
                csv_format = gr.Checkbox(label="CSV file", value=False)
        
        # Full-width row for controls
        with gr.Row():
            process_btn = gr.Button("🚀 Start Processing", variant="primary", size="lg", scale=1)
            stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg", scale=1)
            grid_size_slider = gr.Slider(
                minimum=150, maximum=600, value=300, step=25,
                label="🔍 Preview Size", scale=1
            )
        
        # Real-time Results Gallery
        gr.Markdown("📸 Real-time Results Gallery")
        
        hf_gallery = gr.HTML(
            value=create_hf_style_gallery([]),
            elem_id="gallery-container"
        )
        
        # Event handlers - optimized inline
        folder_input.change(validate_folder, folder_input, folder_status)
        process_btn.click(process_folder_gui, [folder_input, output_input, prompt_template, txt_format, json_format, csv_format], hf_gallery)
        grid_size_slider.change(None, grid_size_slider, js="(size) => document.documentElement.style.setProperty('--gallery-grid-size', size + 'px')")
        stop_btn.click(stop_processing_gui)
    
    return interface


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gradio_app.log'),
            logging.StreamHandler()
        ]
    )
    
    # Get universal allowed paths - simplified
    allowed_paths = ["/"] if Path("/").exists() else [f"{d}:\\" for d in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{d}:\\").exists()]
    
    print(f"🌐 Universal file system access enabled")
    print(f"📁 Allowed paths: {allowed_paths}")
    logging.info(f"Application starting with allowed paths: {allowed_paths}")
    
    # Launch Gradio interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        allowed_paths=allowed_paths
    )