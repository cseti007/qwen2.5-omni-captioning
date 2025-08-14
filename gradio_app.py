"""
Gradio GUI for VLLM Qwen2.5-Omni Video/Image Captioning
Real-time Gallery interface with streaming results display and dynamic resizing
"""

import os
import sys
import gradio as gr
from pathlib import Path
from typing import List, Tuple, Dict, Any
import threading
import time
import logging

# Import existing modules
from main import load_config, create_inference_engine, process_single_media
from vllm_inference import list_all_supported_files, detect_media_type

# Set static paths for direct file serving
gr.set_static_paths(paths=["videos/", "images/", "test_media/", "captions/", "static/"])


class CaptionProcessor:
    def __init__(self):
        self.inference_engine = None
        self.is_processing = False
        self.stop_requested = False
    
    def load_model(self, config):
        """Load the inference model"""
        if self.inference_engine is None:
            self.inference_engine = create_inference_engine(config)
        return self.inference_engine
    
    def process_folder_streaming(self, folder_path: str, output_path: str, prompt_template: str, output_formats: List[str]):
        """
        Generator function for real-time Gallery updates with streaming results
        Yields gallery items as they are processed for immediate display
        """
        if not folder_path or not Path(folder_path).exists():
            yield [("error", "⚠ Invalid folder path!")]
            return
        
        # Load config and override settings
        config = load_config()
        config['paths']['input_dir'] = folder_path
        config['paths']['output_dir'] = output_path  # Override output directory
        config['files']['prompts_config'] = f"./example_prompts/{prompt_template}"  # Override prompt template
        config['conversation']['enable_multi_round'] = True  # Auto-detection enabled
        config['paths']['output_formats'] = output_formats
        
        # Reload prompts with selected template
        config = load_config_with_prompts(config)
        
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
        
        # Initialize gallery results
        gallery_results = []
        
        # Process files one by one and yield results
        for i, media_file in enumerate(media_files):
            if self.stop_requested:
                # Add stop indicator and break
                gallery_results.append((
                    str(media_file),
                    f"ℹ️ STOPPED\n\n{media_file.name}\n\nProcessing stopped by user at {i+1}/{len(media_files)}"
                ))
                yield gallery_results
                break
            
            try:
                media_type = detect_media_type(str(media_file))
                
                # Add processing indicator
                gallery_results.append((
                    str(media_file),
                    f"🔄 PROCESSING ({i+1}/{len(media_files)})\n\n{media_file.name}\n\nType: {media_type}\nGenerating caption..."
                ))
                yield gallery_results
                
                # Process single file
                success = process_single_media(str(media_file), self.inference_engine, config)
                
                if success:
                    # Try to read the generated caption from output file
                    caption_text = self._read_generated_caption(str(media_file), config)
                    
                    # Update with success result
                    gallery_results[-1] = (
                        str(media_file),
                        f"✅ COMPLETED ({i+1}/{len(media_files)})\n\n{media_file.name}\n\nType: {media_type}\n\nCaption:\n{caption_text}"
                    )
                else:
                    # Update with failure result
                    gallery_results[-1] = (
                        str(media_file),
                        f"⚠ FAILED ({i+1}/{len(media_files)})\n\n{media_file.name}\n\nType: {media_type}\n\nFailed to generate caption"
                    )
                
                yield gallery_results
                        
            except Exception as e:
                # Update with error result
                if gallery_results and len(gallery_results) > i:
                    gallery_results[-1] = (
                        str(media_file),
                        f"⚠ ERROR ({i+1}/{len(media_files)})\n\n{media_file.name}\n\nError: {str(e)}"
                    )
                else:
                    gallery_results.append((
                        str(media_file),
                        f"⚠ ERROR ({i+1}/{len(media_files)})\n\n{media_file.name}\n\nError: {str(e)}"
                    ))
                yield gallery_results
        
        # Final summary
        successful = sum(1 for _, caption in gallery_results if "✅ COMPLETED" in caption)
        failed = len(gallery_results) - successful
        
        # Add summary to first item if exists
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
            
            # Try to read caption from txt file (most common format)
            caption_file = output_dir / f"{media_file.stem}.txt"
            if caption_file.exists():
                with open(caption_file, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                    return caption
            
            return "Caption generated but file not found"
            
        except Exception as e:
            return f"Error reading caption: {str(e)}"
    
    def stop_processing(self):
        """Stop the current processing"""
        print(f"🛑 Stop request received at {time.strftime('%H:%M:%S')}")
        logging.info("Stop processing requested by user")
        
        if self.is_processing:
            self.stop_requested = True
            print(f"ℹ️ Processing will stop after current file completes")
            logging.info("Stop flag set - processing will halt after current file")
        else:
            print(f"ℹ️ No active processing to stop")
            logging.info("Stop requested but no active processing found")


def create_hf_style_gallery(gallery_results, grid_size=300):
    """Create HuggingFace-style widget gallery with dynamic resizing support"""
    if not gallery_results:
        return f"""
        <div class='hf-gallery-empty'>No results yet...</div>
        <style>
            :root {{
                --gallery-grid-size: {grid_size}px;
            }}
        </style>
        """
    
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
            
        # Convert path for web serving using Gradio's file handling
        try:
            file_path = Path(path)
            # For absolute paths outside working directory, use direct path
            if file_path.is_absolute():
                web_path = str(file_path).replace('\\', '/')
            else:
                # For relative paths, make relative to current working directory
                web_path = str(file_path.relative_to(Path.cwd())).replace('\\', '/')
        except Exception as e:
            # Fallback: use the path as-is
            web_path = str(path).replace('\\', '/')
        
        # Detect if it's video or image
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
    
    # Enhanced CSS with custom properties and JavaScript for dynamic resizing
    html = f"""
    <div class="hf-gallery-container">
        <div class="hf-gallery" id="hf-gallery">
            {''.join(items_html)}
        </div>
    </div>
    
    <style>
        :root {{
            --gallery-grid-size: {grid_size}px;
            --gallery-gap: 12px;
            --gallery-border-radius: 8px;
        }}
        
        .hf-gallery-container {{
            width: 100%;
            padding: 10px;
        }}
        
        .hf-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(var(--gallery-grid-size), 1fr));
            gap: var(--gallery-gap);
            padding: 0;
        }}
        
        .hf-gallery-item {{
            position: relative;
            background: #1a1a1a;
            border-radius: var(--gallery-border-radius);
            overflow: hidden;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: pointer;
            border: 2px solid transparent;
        }}
        
        .hf-gallery-item:hover {{
            transform: scale(1.02);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            border-color: #007acc;
        }}
        
        .hf-gallery-item img,
        .hf-gallery-item video {{
            width: 100%;
            height: var(--gallery-grid-size);
            object-fit: cover;
            display: block;
        }}
        
        .hf-gallery-item.error {{
            background: #ff4444;
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: var(--gallery-grid-size);
        }}
        
        .play-overlay {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.7);
            color: white;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            opacity: 0.8;
            transition: opacity 0.2s ease;
        }}
        
        .hf-gallery-item:hover .play-overlay {{
            opacity: 1;
        }}
        
        .caption {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.9));
            color: white;
            padding: 20px 12px 12px;
            font-size: 12px;
            line-height: 1.3;
            max-height: 60%;
            overflow-y: auto;
            opacity: 0;
            transition: opacity 0.2s ease;
        }}
        
        .hf-gallery-item:hover .caption {{
            opacity: 1;
        }}
        
        .hf-gallery-empty {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-style: italic;
        }}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {{
            :root {{
                --gallery-grid-size: min(var(--gallery-grid-size), 250px);
            }}
        }}
        
        @media (max-width: 480px) {{
            :root {{
                --gallery-grid-size: min(var(--gallery-grid-size), 200px);
            }}
        }}
    </style>
    
    <script>
        // Initialize gallery resize functionality
        (function() {{
            let resizeTimeout;
            
            // Debounced resize function for smooth performance
            window.resizeGallery = function(size) {{
                clearTimeout(resizeTimeout);
                resizeTimeout = setTimeout(() => {{
                    // Update CSS custom property
                    document.documentElement.style.setProperty('--gallery-grid-size', size + 'px');
                    
                    // Force layout recalculation for immediate visual feedback
                    const gallery = document.getElementById('hf-gallery');
                    if (gallery) {{
                        // Trigger reflow
                        gallery.style.opacity = '0.99';
                        requestAnimationFrame(() => {{
                            gallery.style.opacity = '1';
                        }});
                    }}
                }}, 50); // 50ms debounce for smooth slider interaction
            }};
            
            // Initialize with current grid size
            if (typeof window.currentGridSize !== 'undefined') {{
                window.resizeGallery(window.currentGridSize);
            }}
        }})();
    </script>
    """
    
    return html


# Global processor instance
processor = CaptionProcessor()


def process_folder_gui(folder_path: str, output_path: str, prompt_template: str, txt_format: bool, json_format: bool, csv_format: bool, grid_size: int):
    """
    GUI wrapper for folder processing with real-time HuggingFace-style Gallery streaming
    Returns generator for real-time gallery updates with current grid size
    """
    
    # Determine output formats
    output_formats = []
    if txt_format:
        output_formats.append("txt")
    if json_format:
        output_formats.append("json") 
    if csv_format:
        output_formats.append("csv")
    
    if not output_formats:
        yield create_hf_style_gallery([("error", "⚠ Please select at least one output format!")], grid_size)
        return
    
    # Set processing state
    processor.is_processing = True
    processor.stop_requested = False
    
    try:
        # Use streaming generator with auto-detected rounds
        for gallery_results in processor.process_folder_streaming(
            folder_path, output_path, prompt_template, output_formats
        ):
            # Convert to HuggingFace-style gallery with current grid size
            yield create_hf_style_gallery(gallery_results, grid_size)
    finally:
        processor.is_processing = False


def update_gallery_grid_size(size):
    """
    Update gallery grid size using direct JavaScript execution
    This function is called via the js parameter in Gradio
    """
    # This function won't be called directly, the JS runs in the browser
    pass


def stop_processing_gui():
    """Stop button handler with logging"""
    print(f"\n{'='*50}")
    print(f"🛑 STOP BUTTON PRESSED - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    
    # Log to file as well
    logging.info(f"Stop button clicked by user at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Call the actual stop function
    processor.stop_processing()
    
    print(f"📋 Stop request processed successfully")
    return "ℹ️ Stop requested - processing will complete current file and halt"


def list_prompt_templates():
    """List available prompt TOML files from example_prompts directory"""
    prompts_dir = Path("example_prompts")
    if not prompts_dir.exists():
        return ["prompts.toml"]  # fallback
    
    toml_files = [f.name for f in prompts_dir.glob("*.toml")]
    return sorted(toml_files) if toml_files else ["prompts.toml"]


def load_config_with_prompts(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reload prompts configuration with selected template"""
    import toml
    
    prompts_file = config.get('files', {}).get('prompts_config', 'prompts.toml')
    prompts_path = Path(prompts_file)
    
    if prompts_path.exists():
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts_config = toml.load(f)
        config['prompts'] = prompts_config
    
    return config


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


def validate_folder(folder_path: str):
    """Validate folder path and show file count"""
    if not folder_path:
        return "📁 Please enter a folder path"
    
    path = Path(folder_path)
    if not path.exists():
        return "⚠ Folder does not exist"
    
    if not path.is_dir():
        return "⚠ Path is not a directory"
    
    # Count supported files
    try:
        media_files = list_all_supported_files(folder_path)
        video_count = sum(1 for f in media_files if detect_media_type(str(f)) == "video")
        image_count = len(media_files) - video_count
        
        if not media_files:
            return "⚠️ No supported media files found"
        
        return f"✅ Found {len(media_files)} files ({video_count} videos, {image_count} images)"
    except Exception as e:
        return f"⚠ Error scanning folder: {str(e)}"


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="VLLM Caption Generator", theme=gr.themes.Monochrome(primary_hue="gray").set(
        body_background_fill="*primary_950",
        body_background_fill_dark="*primary_950"
    )) as interface:
        
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
        
        # New full-width row for controls
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
        
        # Process with grid size as input
        process_event = process_btn.click(
            fn=process_folder_gui,
            inputs=[folder_input, output_input, prompt_template, txt_format, json_format, csv_format, grid_size_slider],
            outputs=[hf_gallery]
        )
        
        # Dynamic grid size updating with direct JavaScript execution
        grid_size_slider.change(
            fn=None,  # No Python function
            inputs=[grid_size_slider],
            js="""
            function updateGallerySize(size) {
                // Direct DOM manipulation for immediate response
                const gallery = document.getElementById('hf-gallery');
                if (gallery) {
                    // Update CSS grid template columns
                    gallery.style.gridTemplateColumns = `repeat(auto-fill, minmax(${size}px, 1fr))`;
                    
                    // Update all gallery items size
                    const items = gallery.querySelectorAll('.hf-gallery-item img, .hf-gallery-item video');
                    items.forEach(item => {
                        item.style.height = size + 'px';
                    });
                    
                    // Update CSS custom property for consistency
                    document.documentElement.style.setProperty('--gallery-grid-size', size + 'px');
                    
                    console.log('Gallery resized to:', size + 'px');
                }
                return size;
            }
            """
        )
        
        stop_btn.click(
            fn=stop_processing_gui,
            outputs=[stop_status],
            cancels=[process_event]
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
    
    # Get universal allowed paths for all platforms
    allowed_paths = get_universal_allowed_paths()
    
    print(f"🌐 Universal file system access enabled")
    print(f"📁 Allowed paths: {allowed_paths}")
    logging.info(f"Application starting with allowed paths: {allowed_paths}")
    
    # Launch Gradio interface with universal file access
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        allowed_paths=allowed_paths
    )