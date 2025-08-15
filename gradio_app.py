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

# Set static paths for direct file serving
gr.set_static_paths(paths=["videos/", "images/", "test_media/", "captions/", "static/"])


class CaptionProcessor:
    def __init__(self):
        self.inference_engine = None
        self.is_processing = False
        self.stop_event = threading.Event()
        self.current_thread = None
    
    def load_model(self, config):
        """Load the inference model"""
        if self.inference_engine is None:
            self.inference_engine = create_inference_engine(config)
        return self.inference_engine
    
    def process_folder_streaming(self, folder_path: str, output_path: str, prompt_template: str, output_formats: List[str]):
        """
        Generator function for real-time Gallery updates with streaming results
        """
        if not folder_path or not Path(folder_path).exists():
            yield [("error", "⚠ Invalid folder path!")]
            return
        
        # Load config from file
        config = load_config()

        # Override config values from GUI BEFORE logging
        config['paths']['input_dir'] = folder_path
        config['paths']['output_dir'] = output_path
        config['files']['prompts_config'] = prompt_template  # Fixed: just the filename
        config['conversation']['enable_multi_round'] = True
        config['paths']['output_formats'] = output_formats

        logging.info(f"📋 Using prompt template: {config['files']['prompts_config']}")

        # Load prompts using the overridden config
        config = self._load_config_with_prompts(config)
        
        # Find media files
        media_files = list_all_supported_files(folder_path)
        if not media_files:
            yield [("error", "⚠ No supported media files found in folder!")]
            return
        
        # Load model
        try:
            self.load_model(config)
        except Exception as e:
            yield [("error", f"⚠ Model loading failed: {str(e)}")]
            return
        
        gallery_results = []
        
        # Process files with stop checking
        for i, media_file in enumerate(media_files):
            if self.stop_event.is_set():
                gallery_results.append((
                    str(media_file),
                    f"🛑 STOPPED\n\n{media_file.name}\n\nProcessing stopped by user at {i+1}/{len(media_files)}"
                ))
                yield gallery_results
                break
            
            try:
                media_type = detect_media_type(str(media_file))
                
                gallery_results.append((
                    str(media_file),
                    f"🔄 PROCESSING ({i+1}/{len(media_files)})\n\n{media_file.name}\n\nType: {media_type}\nGenerating caption..."
                ))
                yield gallery_results
                
                if self.stop_event.is_set():
                    gallery_results[-1] = (
                        str(media_file),
                        f"🛑 STOPPED\n\n{media_file.name}\n\nStopped before processing"
                    )
                    yield gallery_results
                    break
                
                success = process_single_media(str(media_file), self.inference_engine, config)
                
                if success:
                    caption_text = self._read_generated_caption(str(media_file), config)
                    gallery_results[-1] = (
                        str(media_file),
                        f"✅ COMPLETED ({i+1}/{len(media_files)})\n\n{media_file.name}\n\nType: {media_type}\n\nCaption:\n{caption_text}"
                    )
                else:
                    gallery_results[-1] = (
                        str(media_file),
                        f"⚠ FAILED ({i+1}/{len(media_files)})\n\n{media_file.name}\n\nType: {media_type}\n\nFailed to generate caption"
                    )
                
                yield gallery_results
                            
            except Exception as e:
                gallery_results[-1] = (
                    str(media_file),
                    f"⚠ ERROR ({i+1}/{len(media_files)})\n\n{media_file.name}\n\nError: {str(e)}"
                )
                yield gallery_results
        
        successful = sum(1 for _, caption in gallery_results if "✅ COMPLETED" in caption)
        failed = len(gallery_results) - successful
        
        if gallery_results:
            first_item = gallery_results[0]
            summary_caption = f"{first_item[1]}\n\n📊 FINAL SUMMARY:\n✅ Successful: {successful}\n⚠ Failed: {failed}"
            gallery_results[0] = (first_item[0], summary_caption)
        
        yield gallery_results
    
    def _read_generated_caption(self, media_path: str, config: Dict[str, Any]) -> str:
        """Read generated caption from output file"""
        try:
            media_file = Path(media_path)
            output_dir = Path(config['paths']['output_dir'])
            caption_file = output_dir / f"{media_file.stem}.txt"
            
            if caption_file.exists():
                with open(caption_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            return "Caption generated but file not found"
        except Exception as e:
            return f"Error reading caption: {str(e)}"
    
    def _load_config_with_prompts(self, config: Dict[str, Any]) -> Dict[str, Any]:
        prompts_file = config.get('files', {}).get('prompts_config', 'prompts.toml')
        
        # Ensure relative path points to example_prompts/
        if not prompts_file.startswith("example_prompts/") and not prompts_file.startswith("./example_prompts/"):
            prompts_file = f"./example_prompts/{prompts_file}"
        
        prompts_path = Path(prompts_file)
        
        if prompts_path.exists():
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_config = toml.load(f)
            config['prompts'] = prompts_config
        else:
            print(f"⚠ Prompt file not found: {prompts_path}")
        
        return config

    
    def start_processing(self):
        """Start processing and clear stop signal"""
        self.is_processing = True
        self.stop_event.clear()
        logging.info("Processing started")
    
    def stop_processing(self):
        """Stop the current processing"""
        print(f"🛑 Stop request received at {time.strftime('%H:%M:%S')}")
        logging.info("Stop processing requested by user")
        
        if self.is_processing:
            self.stop_event.set()
            print(f"ℹ️ Processing will stop after current file completes")
            logging.info("Stop signal sent - processing will halt")
        else:
            print(f"ℹ️ No active processing to stop")
            logging.info("Stop requested but no active processing found")


def create_hf_style_gallery(gallery_results):
    """Create HuggingFace-style widget gallery with clean HTML"""
    if not gallery_results:
        return '<div class="hf-gallery-empty">No results yet...</div>'
    
    items_html = []
    for path, caption in gallery_results:
        if path == "error":
            items_html.append(f"""
            <div class="hf-gallery-item error">
                <div class="error-content">⚠ Error</div>
                <div class="caption">{caption}</div>
            </div>
            """)
            continue
            
        # Convert path for web serving
        try:
            file_path = Path(path)
            web_path = str(file_path).replace('\\', '/') if file_path.is_absolute() else str(file_path.relative_to(Path.cwd())).replace('\\', '/')
        except Exception:
            web_path = str(path).replace('\\', '/')
        
        # Check if video or image
        is_video = Path(path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if is_video:
            items_html.append(f"""
            <div class="hf-gallery-item video" 
                 onmouseover="this.querySelector('video').play().catch(e=>console.log('Video play failed:', e))" 
                 onmouseout="const v=this.querySelector('video'); v.pause(); v.currentTime=0;"
                 onclick="const v=this.querySelector('video'); if(v.paused) v.play(); else v.pause();">
                <video muted loop preload="auto">
                    <source src="/gradio_api/file={web_path}" type="video/mp4">
                    Your browser does not support video playbook.
                </source>
                </video>
                <div class="play-overlay">▶</div>
                <div class="caption">{caption}</div>
            </div>
            """)
        else:
            items_html.append(f"""
            <div class="hf-gallery-item image" onclick="const img=this.querySelector('img'); window.open(img.src, '_blank');">
                <img src="/gradio_api/file={web_path}" alt="Generated image" loading="lazy">
                <div class="caption">{caption}</div>
            </div>
            """)
    
    return f"""
    <div class="hf-gallery-container">
        <div class="hf-gallery" id="hf-gallery">
            {''.join(items_html)}
        </div>
    </div>
    """


def get_universal_allowed_paths():
    """Get universal allowed paths for all platforms (Windows, Linux, macOS)"""
    allowed_paths = []
    
    # Linux/macOS root
    if Path("/").exists():
        allowed_paths.append("/")
    
    # Windows drives
    for drive in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive_path = f"{drive}:\\"
        if Path(drive_path).exists():
            allowed_paths.append(drive_path)
    
    return allowed_paths


def list_prompt_templates():
    """List available prompt TOML files from example_prompts directory"""
    prompts_dir = Path("example_prompts")
    if not prompts_dir.exists():
        return ["prompts.toml"]
    
    toml_files = [f.name for f in prompts_dir.glob("*.toml")]
    return sorted(toml_files) if toml_files else ["prompts.toml"]


def validate_folder(folder_path: str):
    """Validate folder path and show file count"""
    if not folder_path:
        return "📁 Please enter a folder path"
    
    path = Path(folder_path)
    if not path.exists():
        return "⚠ Folder does not exist"
    
    if not path.is_dir():
        return "⚠ Path is not a directory"
    
    try:
        media_files = list_all_supported_files(folder_path)
        video_count = sum(1 for f in media_files if detect_media_type(str(f)) == "video")
        image_count = len(media_files) - video_count
        
        if not media_files:
            return "⚠️ No supported media files found"
        
        return f"✅ Found {len(media_files)} files ({video_count} videos, {image_count} images)"
    except Exception as e:
        return f"⚠ Error scanning folder: {str(e)}"


# Global processor instance
processor = CaptionProcessor()


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
        yield create_hf_style_gallery([("error", "⚠ Please select at least one output format!")])
        return
    
    # Start processing
    processor.start_processing()
    
    try:
        for gallery_results in processor.process_folder_streaming(
            folder_path, output_path, prompt_template, output_formats
        ):
            yield create_hf_style_gallery(gallery_results)
    finally:
        processor.is_processing = False


def stop_processing_gui():
    """Stop button handler with logging"""
    print(f"\n{'='*50}")
    print(f"🛑 STOP BUTTON PRESSED - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    logging.info(f"Stop button clicked by user at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    processor.stop_processing()
    
    print(f"📋 Stop request processed successfully")
    return "ℹ️ Stop requested - processing will complete current file and halt"


def create_interface():
    with gr.Blocks(
        title="VLLM Caption Generator", 
        css_paths=["static/gallery.css"],
        js="""
        () => {
            if (!window.location.href.includes('__theme=dark')) {
                window.location.href += '?__theme=dark';
            }
        }
        """,
        theme=gr.themes.Monochrome(primary_hue="gray").set(
            body_background_fill="*primary_950",
            body_background_fill_dark="*primary_950"
        )
    ) as interface:
        
        gr.Markdown("# 🎯 VLLM Qwen2.5-Omni Caption Generator")
        gr.Markdown("Generate captions for images and videos using VLLM with real-time progress display")
        
        with gr.Row():
            # Left column - main inputs
            with gr.Column(scale=4):
                with gr.Group():
                    # Row 1: Input folder + status
                    with gr.Row():
                        folder_input = gr.Textbox(
                            label="📁 Input Folder Path",
                            placeholder="/path/to/your/media/folder",
                            value=str(Path.cwd() / "images"),
                            scale=3
                        )
                        
                        folder_status = gr.Textbox(
                            label="📊 Folder Status",
                            value="📁 Please enter a folder path",
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
            stop_status = gr.Textbox(
                value="Control Status: Ready to process",
                interactive=False,
                show_label=False,
                container=False,
                scale=1
            )
        
        # Real-time Results Gallery with size control
        with gr.Row():
            gr.Markdown("📸 Real-time Results Gallery")
            grid_size_slider = gr.Slider(
                minimum=150, maximum=600, value=300, step=25,
                label="🔍 Preview Size", scale=1
            )
        
        hf_gallery = gr.HTML(
            value=create_hf_style_gallery([]),
            elem_id="gallery-container"
        )
        
        # Event handlers
        folder_input.change(
            fn=validate_folder,
            inputs=[folder_input],
            outputs=[folder_status]
        )
        
        # Process button - start processing
        process_btn.click(
            fn=process_folder_gui,
            inputs=[folder_input, output_input, prompt_template, txt_format, json_format, csv_format],
            outputs=[hf_gallery]
        )
        
        # Dynamic grid size updating
        grid_size_slider.change(
            fn=None,
            inputs=[grid_size_slider],
            js="(size) => document.documentElement.style.setProperty('--gallery-grid-size', size + 'px')"
        )
        
        # Stop button - fixed implementation
        stop_btn.click(
            fn=stop_processing_gui,
            outputs=[stop_status]
        )
    
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
    
    # Get universal allowed paths for full system access
    allowed_paths = get_universal_allowed_paths()
    
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