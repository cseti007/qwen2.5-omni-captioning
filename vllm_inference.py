"""
VLLM Inference Module for Qwen2.5-Omni Video/Image Captioning
Refactored version with conversation history support
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from vllm import LLM, SamplingParams
from output_writer import save_conversation_round, save_conversation_final


class ConversationManager:
    """Manages conversation history for multi-round conversations"""
    
    def __init__(self):
        self.histories: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_history(self, media_path: str) -> List[Dict[str, Any]]:
        """Get conversation history for a media file"""
        return self.histories.get(media_path, [])
    
    def add_message(self, media_path: str, role: str, content: str):
        """Add a message to conversation history"""
        if media_path not in self.histories:
            self.histories[media_path] = []
        self.histories[media_path].append({"role": role, "content": content})
    
    def get_previous_caption(self, media_path: str) -> str:
        """Get the last assistant message from conversation history"""
        history = self.histories.get(media_path, [])
        return history[-1]["content"] if history and history[-1]["role"] == "assistant" else ""
    
    def clear(self):
        """Clear all conversation histories"""
        self.histories.clear()


class ChatMLBuilder:
    """Builds ChatML prompts from conversation history"""
    
    @staticmethod
    def build_prompt(conversation: List[Dict[str, Any]], prompt_mode: str, media_type: str) -> str:
        """Build ChatML prompt string from conversation history"""
        prompt_parts = []
        
        for message in conversation:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                if prompt_mode == "text":
                    prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
                else:
                    # Add media tokens for multimodal mode
                    media_token = "<|vision_bos|><|VIDEO|><|vision_eos|>" if media_type == "video" else "<|vision_bos|><|IMAGE|><|vision_eos|>"
                    prompt_parts.append(f"<|im_start|>user\n{media_token}{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Add generation prompt
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)


def detect_media_type(file_path: str) -> str:
    """Auto-detect media type from file extension"""
    extension = Path(file_path).suffix.lower()
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    if extension in video_extensions:
        return "video"
    elif extension in image_extensions:
        return "image"
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def list_all_supported_files(directory: str) -> List[Path]:
    """Find all supported media files in directory"""
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    all_exts = video_exts | image_exts
    
    files = []
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in all_exts:
            files.append(file_path)
    
    return sorted(files)


class MediaLoader:
    """Handles loading and preprocessing of media files"""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image file as numpy array"""
        image = Image.open(image_path).convert('RGB')
        return np.array(image)
    
    @staticmethod
    def load_video(video_path: str, num_frames: int = 16) -> np.ndarray:
        """Load video file as numpy array"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Calculate frame indices
        if num_frames >= total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames extracted from: {video_path}")
        
        return np.array(frames)


class PromptProcessor:
    """Processes and prepares prompts for different rounds"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def get_round_prompts(self, round_num: int) -> Dict[str, str]:
        """Get prompts for a specific round (media type auto-detected)"""
        round_key = f"round{round_num}"
        prompts_config = self.config['prompts']
        
        if round_key in prompts_config:
            return prompts_config[round_key]
        else:
            raise ValueError(f"No prompts configured for {round_key}")
    
    def process_prompts(self, system_prompt: str, user_prompt: str, previous_caption: str = "") -> tuple[str, str]:
        """Process prompts with trigger words and previous caption"""
        # Apply trigger word
        trigger_word = self.config['prompts'].get('general', {}).get('trigger_word', '')
        if trigger_word:
            system_prompt = system_prompt.replace('{trigger_word}', trigger_word)
            user_prompt = user_prompt.replace('{trigger_word}', trigger_word)
        
        # Apply previous caption if available
        if previous_caption and "{previous_caption}" in user_prompt:
            user_prompt = user_prompt.format(previous_caption=previous_caption)
        
        return system_prompt, user_prompt
    
    def get_available_rounds(self) -> int:
        """Auto-detect available rounds from prompts configuration"""
        prompts_config = self.config['prompts']
        i = 1
        while f"round{i}" in prompts_config:
            i += 1
        return i - 1


class VLLMInference:
    """Main VLLM inference handler for Qwen2.5-Omni"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        
        # Initialize components
        self.conversation_manager = ConversationManager()
        self.prompt_processor = PromptProcessor(config)
        self.media_loader = MediaLoader()
        self.chatml_builder = ChatMLBuilder()
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize VLLM model"""
        logging.getLogger("vllm").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        
        model_config = self.config['model']
        hardware_config = self.config['hardware']
        generation_config = self.config['generation']
        processing_config = self.config['processing']
        
        # Set engine version based on config
        vllm_version = hardware_config.get('vllm_engine', 'v0')
        os.environ['VLLM_USE_V1'] = '1' if vllm_version.lower() == 'v1' else '0'
        logging.info(f"Model loading: {model_config['name']} (VLLM {vllm_version.upper()})")
        
        # Build VLLM parameters
        llm_params = {
            "model": model_config['name'],
            "trust_remote_code": model_config['trust_remote_code'],
            "dtype": model_config['dtype'],
            "max_model_len": model_config['max_model_len'],
            "gpu_memory_utilization": hardware_config['gpu_memory_utilization'],
            "tensor_parallel_size": hardware_config['tensor_parallel_size'],
            "limit_mm_per_prompt": {"video": 1, "audio": 1, "image": 1},
            "mm_processor_kwargs": {"fps": processing_config.get('fps', 2.0)},
            "disable_log_stats": False,          # Stats logging
        }
        
        # Add V1 optimizations
        if vllm_version.lower() == 'v1':
            llm_params["max_num_batched_tokens"] = processing_config.get('max_num_batched_tokens', 16384)
        
        self.llm = LLM(**llm_params)
        self.sampling_params = SamplingParams(
            temperature=generation_config['temperature'],
            max_tokens=generation_config['max_tokens'],
            top_p=generation_config['top_p'],
        )
    
    def get_effective_batch_size(self, media_type: str) -> int:
        """Get effective batch size based on media type"""
        return self.config['processing'].get('batch_size', 1)
    
    def generate_caption(self, media_path: str, media_type: str = "video") -> str:
        """Generate caption for a single media file with multi-round support"""
        conversation_config = self.config.get('conversation', {})
        enable_multi_round = conversation_config.get('enable_multi_round', False)
        rounds = self.prompt_processor.get_available_rounds() if enable_multi_round else 1
        
        # Load media data
        media_data = self._load_media_data(media_path, media_type)
        caption = ""
        
        # Execute conversation rounds
        for round_num in range(1, rounds + 1):
            if hasattr(self.llm, 'clear_cache'):
                self.llm.clear_cache()
            
            caption = self._execute_single_round(media_data, round_num, media_path, media_type)
            
            print(f"\n🎯 ROUND {round_num} CAPTION - {Path(media_path).name} 🎯")
            print(caption)
            print("="*50)
        
        save_conversation_final(caption, media_path, self.config)
        return caption
    
    def generate_batch_captions(self, media_paths: List[str], media_type: str = "image") -> List[str]:
        """Generate captions for multiple media files with multi-round support"""
        if not media_paths:
            return []
        
        conversation_config = self.config.get('conversation', {})
        enable_multi_round = conversation_config.get('enable_multi_round', False)
        rounds = self.prompt_processor.get_available_rounds() if enable_multi_round else 1
        
        # Load all media data
        media_data_list = [self._load_media_data(path, media_type) for path in media_paths]
        captions = [""] * len(media_paths)
        
        # Execute conversation rounds
        for round_num in range(1, rounds + 1):
            captions = self._execute_batch_round(media_data_list, round_num, media_paths, media_type)
            
            for i, caption in enumerate(captions):
                print(f"\n🎯 BATCH ROUND {round_num} - FILE {i+1}/{len(media_paths)} 🎯")
                print(f"File: {Path(media_paths[i]).name}")
                print(caption)
                print("="*80)
        
        # Save final results
        if rounds > 1:
            for caption, media_path in zip(captions, media_paths):
                save_conversation_final(caption, media_path, self.config)
        
        return captions
    
    def _load_media_data(self, media_path: str, media_type: str) -> np.ndarray:
        """Load media data based on type"""
        if media_type == "video":
            return self.media_loader.load_video(media_path)
        elif media_type == "image":
            return self.media_loader.load_image(media_path)
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
    
    def _execute_single_round(self, media_data: np.ndarray, round_num: int, 
                            media_path: str, media_type: str) -> str:
        """Execute a single conversation round for one media file"""
        # Get and process prompts
        round_prompts = self.prompt_processor.get_round_prompts(round_num)
        prompt_mode = round_prompts.get('mode', 'multimodal')
        previous_caption = self.conversation_manager.get_previous_caption(media_path)
        system_prompt, user_prompt = self.prompt_processor.process_prompts(
            round_prompts['system_prompt'], 
            round_prompts['user_prompt'], 
            previous_caption
        )
        
        # Build conversation and generate
        conversation = self._build_conversation(media_path, round_num, system_prompt, user_prompt, prompt_mode)
        
        query_inputs = self._create_query_inputs(conversation, prompt_mode, media_type, media_data)
        caption = self._generate_with_llm(query_inputs, media_path)
        
        # Update conversation history
        self.conversation_manager.add_message(media_path, "user", user_prompt)
        self.conversation_manager.add_message(media_path, "assistant", caption)
        
        # Save conversation round
        if self.config.get('processing', {}).get('save_conversations', False):
            save_conversation_round(round_num, system_prompt, user_prompt, caption, media_path, self.config)
        
        return caption
    
    def _execute_batch_round(self, media_data_list: List[np.ndarray], round_num: int,
                           media_paths: List[str], media_type: str) -> List[str]:
        """Execute a single conversation round for batch of media files"""
        round_prompts = self.prompt_processor.get_round_prompts(round_num)
        prompt_mode = round_prompts.get('mode', 'multimodal')
        
        # Create batch inputs
        batch_inputs = []
        processed_prompts = []
        
        for i, media_path in enumerate(media_paths):
            previous_caption = self.conversation_manager.get_previous_caption(media_path)
            system_prompt, user_prompt = self.prompt_processor.process_prompts(
                round_prompts['system_prompt'],
                round_prompts['user_prompt'],
                previous_caption
            )
            processed_prompts.append((system_prompt, user_prompt))
            
            # Build conversation with system prompt fix
            conversation = self._build_conversation(media_path, round_num, system_prompt, user_prompt, prompt_mode)
            query_inputs = self._create_query_inputs(conversation, prompt_mode, media_type, media_data_list[i])
            batch_inputs.append(query_inputs)
        
        # Generate batch responses
        outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
        
        captions = []
        for i, (output, media_path) in enumerate(zip(outputs, media_paths)):
            if output and len(output.outputs) > 0:
                caption = output.outputs[0].text.strip()
                
                # Update conversation history
                _, user_prompt = processed_prompts[i]
                self.conversation_manager.add_message(media_path, "user", user_prompt)
                self.conversation_manager.add_message(media_path, "assistant", caption)
                
                # Save conversation round
                if self.config.get('processing', {}).get('save_conversations', False):
                    system_prompt, _ = processed_prompts[i]
                    save_conversation_round(round_num, system_prompt, user_prompt, caption, media_path, self.config)
                
                captions.append(caption)
            else:
                logging.warning(f"Empty caption generated for {Path(media_path).name}")
                captions.append("")
        
        return captions
    
    def _build_conversation(self, media_path: str, round_num: int, 
                          system_prompt: str, user_prompt: str, prompt_mode: str) -> List[Dict[str, Any]]:
        """Build conversation history for a round"""
        # Always start with current round's system prompt
        conversation = [{"role": "system", "content": system_prompt}]
        
        # Add previous assistant responses ONLY in text mode
        if round_num > 1 and prompt_mode == "text":
            history = self.conversation_manager.get_history(media_path)
            for msg in history:
                # Only add assistant responses (captions from previous rounds)
                if msg["role"] == "assistant":
                    conversation.append(msg)
        
        # Add current user prompt
        conversation.append({"role": "user", "content": user_prompt})
        return conversation
    
    def _create_query_inputs(self, conversation: List[Dict[str, Any]], prompt_mode: str,
                           media_type: str, media_data: np.ndarray) -> Dict[str, Any]:
        """Create query inputs for VLLM generate API"""
        prompt = self.chatml_builder.build_prompt(conversation, prompt_mode, media_type)
        
        if prompt_mode == "text":
            return {"prompt": prompt}
        else:
            # Multimodal mode
            mm_data = {"video": media_data} if media_type == "video" else {"image": media_data}
            query_inputs = {"prompt": prompt, "multi_modal_data": mm_data}
            
            # Add audio processing for video if enabled
            if (media_type == "video" and 
                self.config['processing'].get('use_audio_in_video', False)):
                query_inputs["mm_processor_kwargs"] = {"use_audio_in_video": True}
            
            return query_inputs
    
    def _generate_with_llm(self, query_inputs: Dict[str, Any], media_path: str) -> str:
        """Generate caption using VLLM"""
        outputs = self.llm.generate(query_inputs, sampling_params=self.sampling_params)
        
        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text.strip()
        else:
            raise RuntimeError("No output generated")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'llm') and self.llm:
            del self.llm
        self.conversation_manager.clear()


def create_inference_engine(config: Dict[str, Any]) -> VLLMInference:
    """Factory function to create VLLM inference engine"""
    return VLLMInference(config)