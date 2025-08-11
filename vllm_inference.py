"""
VLLM Inference Module for Qwen2.5-Omni Video/Image Captioning
Handles model loading and video/image caption generation using official Qwen2.5-Omni pattern
"""

import os
import logging
from typing import Dict, Any, Optional, NamedTuple, List
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from vllm import LLM, SamplingParams


class QueryResult(NamedTuple):
    """Data structure for Qwen2.5-Omni queries"""
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


class VLLMInference:
    """VLLM inference handler for Qwen2.5-Omni"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm: Optional[LLM] = None
        self.sampling_params: Optional[SamplingParams] = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize VLLM model using official Qwen2.5-Omni pattern"""
        model_config = self.config['model']
        hardware_config = self.config['hardware']
        generation_config = self.config['generation']
        processing_config = self.config['processing'] 
        
        logging.info(f"Loading model: {model_config['name']}")
        
        # Set environment variable for V0 engine (required for Omni)
        os.environ['VLLM_USE_V1'] = '0'
        
        # Initialize LLM directly (following official pattern)
        self.llm = LLM(
            model=model_config['name'],
            trust_remote_code=model_config['trust_remote_code'],
            dtype=model_config['dtype'],
            max_model_len=model_config['max_model_len'],
            gpu_memory_utilization=hardware_config['gpu_memory_utilization'],
            tensor_parallel_size=hardware_config['tensor_parallel_size'],
            limit_mm_per_prompt={"video": 1, "audio": 1, "image": 1},
            mm_processor_kwargs={
                "fps": processing_config.get('fps', 2.0),
    },
        )
        
        # Initialize sampling parameters
        self.sampling_params = SamplingParams(
            temperature=generation_config['temperature'],
            max_tokens=generation_config['max_tokens'],
            top_p=generation_config['top_p']
        )
        
        logging.info("Model loaded successfully")
    
    def get_effective_batch_size(self, media_type: str) -> int:
        """Get effective batch size based on media type"""
        config_batch_size = self.config['processing'].get('batch_size', 1)
        
        if media_type == "video":
            # Videos are larger, use smaller batch
            return max(1, config_batch_size // 2)
        else:  # image
            return config_batch_size
    
    def generate_batch_captions(self, media_paths: List[str], media_type: str = "image") -> List[str]:
        """
        Generate captions for multiple media files simultaneously
        
        Args:
            media_paths: List of paths to media files
            media_type: Type of media ("video" or "image")
            
        Returns:
            List of generated captions (one per media file)
        """
        if not self.llm or not self.sampling_params:
            raise RuntimeError("Model not initialized")
        
        if not media_paths:
            return []
            
        conversation_config = self.config.get('conversation', {})
        enable_multi_round = conversation_config.get('enable_multi_round', False)
        rounds = conversation_config.get('rounds', 1)
        
        if not enable_multi_round:
            rounds = 1
        
        # Load all media files
        media_data_list = []
        for media_path in media_paths:
            if media_type == "video":
                media_data = self._load_video_as_numpy(media_path)
            elif media_type == "image":
                media_data = self._load_image_as_numpy(media_path)
            else:
                raise ValueError(f"Unsupported media type: {media_type}")
            media_data_list.append(media_data)
        
        captions = [""] * len(media_paths)
        
        # Execute conversation rounds for all media
        for round_num in range(1, rounds + 1):
            logging.info(f"Starting batch round {round_num}/{rounds} for {len(media_paths)} {media_type} files")
            
            if round_num == 1:
                # First round: Initial descriptions
                captions = self._execute_batch_round(media_data_list, round_num, media_paths, media_type)
            else:
                # Subsequent rounds: Refine previous captions
                captions = self._execute_batch_round(media_data_list, round_num, media_paths, media_type, captions)
            
            logging.info(f"Batch round {round_num} completed")
            for i, caption in enumerate(captions):
                print(f"\n🎯 BATCH ROUND {round_num} - FILE {i+1}/{len(media_paths)} 🎯")
                print(f"File: {Path(media_paths[i]).name}")
                print(caption)
                print("="*80)

        return captions

    def _execute_batch_round(self, media_data_list: List[np.ndarray], round_num: int, 
                           media_paths: List[str], media_type: str, 
                           previous_captions: List[str] = None) -> List[str]:
        """
        Execute a single conversation round for batch of media
        
        Args:
            media_data_list: List of media data as numpy arrays
            round_num: Current round number (1-based)
            media_paths: List of media file paths
            media_type: Type of media ("video" or "image")
            previous_captions: Previous captions for refinement (None for first round)
            
        Returns:
            Generated captions for this round
        """
        processing_config = self.config['processing']
        use_audio_in_video = processing_config.get('use_audio_in_video', False) and media_type == "video"
        
        # Get prompts for this round and media type
        round_key = f"round{round_num}"
        prompts_config = self.config['prompts']
        
        # Check if media-specific prompts exist, fallback to generic if not
        if media_type in prompts_config and round_key in prompts_config[media_type]:
            round_prompts = prompts_config[media_type][round_key]
        elif round_key in prompts_config:
            # Fallback to generic prompts (backward compatibility)
            round_prompts = prompts_config[round_key]
        else:
            raise ValueError(f"No prompts configured for {media_type}.{round_key}")
        
        system_prompt = round_prompts['system_prompt']
        user_prompt = round_prompts['user_prompt']
        
        # Apply trigger_word to prompts if it exists
        trigger_word = prompts_config.get('general', {}).get('trigger_word', '')
        if trigger_word:
            system_prompt = system_prompt.replace('{trigger_word}', trigger_word)
            user_prompt = user_prompt.replace('{trigger_word}', trigger_word)
        
        # Create batch queries
        batch_inputs = []
        for i, media_data in enumerate(media_data_list):
            # For rounds > 1, inject previous caption into user prompt
            current_user_prompt = user_prompt
            if round_num > 1 and previous_captions and i < len(previous_captions):
                current_user_prompt = user_prompt.format(previous_caption=previous_captions[i])
            
            # Create query for this media
            query_result = self._create_media_query(
                question=current_user_prompt,
                system_prompt=system_prompt,
                media_data=media_data,
                media_type=media_type,
                use_audio_in_video=use_audio_in_video
            )
            batch_inputs.append(query_result.inputs)
        
        # Generate batch response
        try:
            outputs = self.llm.generate(
                batch_inputs, 
                sampling_params=self.sampling_params
            )
            
            captions = []
            for output in outputs:
                if output and len(output.outputs) > 0:
                    caption = output.outputs[0].text.strip()
                    captions.append(caption)
                else:
                    captions.append("")  # Empty caption for failed generation
                    
            return captions
                    
        except Exception as e:
            logging.error(f"Failed to generate batch captions for round {round_num}: {e}")
            raise

    def generate_caption(self, media_path: str, media_type: str = "video") -> str:
            """
            Generate caption for a video or image file with multi-round conversation support
            
            Args:
                media_path: Path to media file (video or image)
                media_type: Type of media ("video" or "image")
                
            Returns:
                Generated caption text (final result after all rounds)
            """
            if not self.llm or not self.sampling_params:
                raise RuntimeError("Model not initialized")
            
            conversation_config = self.config.get('conversation', {})
            enable_multi_round = conversation_config.get('enable_multi_round', False)
            rounds = conversation_config.get('rounds', 1)
            
            if not enable_multi_round:
                rounds = 1
            
            # Load media once for all rounds
            if media_type == "video":
                media_data = self._load_video_as_numpy(media_path)
            elif media_type == "image":
                media_data = self._load_image_as_numpy(media_path)
            else:
                raise ValueError(f"Unsupported media type: {media_type}")
                
            caption = ""
            
            # Execute conversation rounds
            for round_num in range(1, rounds + 1):
                if round_num == 1:
                    logging.info(f"Starting round {round_num}/{rounds} — loading {media_type} from: {media_path}")
                    # First round: Initial description
                    caption = self._execute_round(media_data, round_num, media_path, media_type)
                else:
                    logging.info(f"Starting round {round_num}/{rounds} — refining previous caption, no {media_type} reload")
                    # Subsequent rounds: Refine previous caption
                    caption = self._execute_round(media_data, round_num, media_path, media_type, caption)
                
                logging.info(f"Round {round_num} completed ({len(caption)} chars)")
                logging.info(f"Caption after round {round_num}:\n{caption}")

            return caption

    def _execute_round(self, media_data: np.ndarray, round_num: int, media_path: str, media_type: str, previous_caption: str = "") -> str:
            """
            Execute a single conversation round
            
            Args:
                media_data: Media data as numpy array (video frames or single image)
                round_num: Current round number (1-based)
                media_path: Path to media file
                media_type: Type of media ("video" or "image")
                previous_caption: Caption from previous round (empty for first round)
                
            Returns:
                Generated caption for this round
            """
            processing_config = self.config['processing']
            use_audio_in_video = processing_config.get('use_audio_in_video', False) and media_type == "video"
            
            # Get prompts for this round and media type
            round_key = f"round{round_num}"
            prompts_config = self.config['prompts']
            
            # Check if media-specific prompts exist, fallback to generic if not
            if media_type in prompts_config and round_key in prompts_config[media_type]:
                round_prompts = prompts_config[media_type][round_key]
            elif round_key in prompts_config:
                # Fallback to generic prompts (backward compatibility)
                round_prompts = prompts_config[round_key]
            else:
                raise ValueError(f"No prompts configured for {media_type}.{round_key}")
            
            system_prompt = round_prompts['system_prompt']
            user_prompt = round_prompts['user_prompt']
            
            # Apply trigger_word to prompts if it exists
            trigger_word = prompts_config.get('general', {}).get('trigger_word', '')
            if trigger_word:
                system_prompt = system_prompt.replace('{trigger_word}', trigger_word)
                user_prompt = user_prompt.replace('{trigger_word}', trigger_word)
            
            # For rounds > 1, inject previous caption into user prompt
            if round_num > 1 and previous_caption:
                user_prompt = user_prompt.format(previous_caption=previous_caption)
            
            # Create query for this round
            query_result = self._create_media_query(
                question=user_prompt,
                system_prompt=system_prompt,
                media_data=media_data,
                media_type=media_type,
                use_audio_in_video=use_audio_in_video
            )
            
            # Generate response
            try:
                outputs = self.llm.generate(
                    query_result.inputs, 
                    sampling_params=self.sampling_params
                )
                
                if outputs and len(outputs) > 0:
                    caption = outputs[0].outputs[0].text.strip()
                    return caption
                else:
                    raise RuntimeError(f"No output generated for round {round_num}")
                    
            except Exception as e:
                logging.error(f"Failed to generate caption for round {round_num}: {e}")
                raise
    
    def _load_image_as_numpy(self, image_path: str) -> np.ndarray:
        """
        Load image file as numpy array (following official pattern)
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array with shape (height, width, channels)
        """
        try:
            # Load image using PIL
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            logging.info(f"Loaded image with shape: {image_array.shape}")
            
            return image_array
            
        except Exception as e:
            logging.error(f"Failed to load image: {e}")
            raise
         
    def _load_video_as_numpy(self, video_path: str, num_frames: int = 16) -> np.ndarray:
        """
        Load video file as numpy array (following official pattern)
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            Video frames as numpy array with shape (num_frames, height, width, channels)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"Video has no frames: {video_path}")
            
            # Calculate frame indices to sample
            if num_frames >= total_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            
            if not frames:
                raise ValueError(f"No frames could be extracted from: {video_path}")
            
            # Convert to numpy array
            video_array = np.array(frames)
            logging.info(f"Loaded video with shape: {video_array.shape}")
            
            return video_array
            
        except Exception as e:
            logging.error(f"Failed to load video: {e}")
            raise
    
    def _create_media_query(
        self, 
        question: str, 
        system_prompt: str,
        media_data: np.ndarray,
        media_type: str,
        use_audio_in_video: bool = False
    ) -> QueryResult:
        """
        Create media query using official Qwen2.5-Omni pattern for video or image
        
        Args:
            question: User question/prompt
            system_prompt: System prompt
            media_data: Media data as numpy array
            media_type: Type of media ("video" or "image")
            use_audio_in_video: Whether to process audio from video (ignored for images)
            
        Returns:
            QueryResult with properly formatted query
        """
        # Create prompt using official format
        if media_type == "video":
            media_token = "<|VIDEO|>"
        elif media_type == "image":
            media_token = "<|IMAGE|>"
        else:
            raise ValueError(f"Unsupported media type: {media_type}")
            
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_bos|>{media_token}<|vision_eos|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # Prepare multimodal data
        if media_type == "video":
            mm_data = {
                "video": media_data,  # NumPy array for video frames
            }
        else:  # image
            mm_data = {
                "image": media_data,  # NumPy array for single image
            }
        
        # Prepare inputs
        inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        
        # Add processor kwargs if needed (only for video)
        if media_type == "video" and use_audio_in_video:
            inputs["mm_processor_kwargs"] = {
                "use_audio_in_video": True,
            }
            logging.info("Audio processing from video enabled")
        
        # Set limits
        if media_type == "video":
            limit_mm_per_prompt = {"video": 1}
            if use_audio_in_video:
                limit_mm_per_prompt["audio"] = 1
        else:  # image
            limit_mm_per_prompt = {"image": 1}
        
        return QueryResult(
            inputs=inputs,
            limit_mm_per_prompt=limit_mm_per_prompt,
        )
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'llm') and self.llm:
            del self.llm
        logging.info("Model cleanup completed")


def create_inference_engine(config: Dict[str, Any]) -> VLLMInference:
    """
    Factory function to create VLLM inference engine
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VLLMInference instance
    """
    return VLLMInference(config)