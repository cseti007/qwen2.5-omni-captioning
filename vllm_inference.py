"""
VLLM Inference Module for Qwen2.5-Omni Video Captioning
Handles model loading and video caption generation using official Qwen2.5-Omni pattern
"""

import os
import logging
from typing import Dict, Any, Optional, NamedTuple
import numpy as np
from PIL import Image
import cv2
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
        
        logging.info(f"Loading model: {model_config['name']}")
        
        # Set environment variable for V0 engine (required for Omni)
        os.environ['VLLM_USE_V1'] = '0'
        
        # Initialize LLM directly (following official pattern)
        self.llm = LLM(
            model=model_config['name'],
            trust_remote_code=model_config['trust_remote_code'],
            dtype=model_config['dtype'],
            max_model_len=model_config['max_model_len'],
            max_num_seqs=hardware_config.get('max_num_seqs', 5),
            gpu_memory_utilization=hardware_config['gpu_memory_utilization'],
            tensor_parallel_size=hardware_config['tensor_parallel_size'],
            limit_mm_per_prompt={"video": 1, "audio": 1},
        )
        
        # Initialize sampling parameters
        self.sampling_params = SamplingParams(
            temperature=generation_config['temperature'],
            max_tokens=generation_config['max_tokens'],
            top_p=generation_config['top_p']
        )
        
        logging.info("Model loaded successfully")
    
    def generate_caption(self, video_path: str) -> str:
        """
        Generate caption for a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Generated caption text
        """
        if not self.llm or not self.sampling_params:
            raise RuntimeError("Model not initialized")
        
        prompts_config = self.config['prompts']
        processing_config = self.config['processing']
        use_audio_in_video = processing_config.get('use_audio_in_video', False)
        
        # Load video as numpy array (following official pattern)
        video_frames = self._load_video_as_numpy(video_path)
        
        # Create query result using official pattern
        query_result = self._create_video_query(
            question=prompts_config['user_prompt'],
            system_prompt=prompts_config['system_prompt'],
            video_frames=video_frames,
            use_audio_in_video=use_audio_in_video
        )
        
        # Generate caption using official pattern
        try:
            logging.info(f"Generating caption for: {video_path}")
            logging.info(f"Video frames shape: {video_frames.shape}")
            
            outputs = self.llm.generate(
                query_result.inputs, 
                sampling_params=self.sampling_params
            )
            
            if outputs and len(outputs) > 0:
                caption = outputs[0].outputs[0].text.strip()
                logging.info(f"Generated caption ({len(caption)} chars)")
                logging.info(f"Caption: {caption}")
                return caption
            else:
                raise RuntimeError("No output generated")
                
        except Exception as e:
            logging.error(f"Failed to generate caption: {e}")
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
    
    def _create_video_query(
        self, 
        question: str, 
        system_prompt: str,
        video_frames: np.ndarray,
        use_audio_in_video: bool = False
    ) -> QueryResult:
        """
        Create video query using official Qwen2.5-Omni pattern
        
        Args:
            question: User question/prompt
            system_prompt: System prompt
            video_frames: Video frames as numpy array
            use_audio_in_video: Whether to process audio from video
            
        Returns:
            QueryResult with properly formatted query
        """
        # Create prompt using official format
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # Prepare multimodal data
        mm_data = {
            "video": video_frames,  # NumPy array, not file path!
        }
        
        # Prepare inputs
        inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        
        # Add processor kwargs if needed
        if use_audio_in_video:
            inputs["mm_processor_kwargs"] = {
                "use_audio_in_video": True,
            }
            logging.info("Audio processing from video enabled")
        
        # Set limits
        limit_mm_per_prompt = {"video": 1}
        if use_audio_in_video:
            limit_mm_per_prompt["audio"] = 1
        
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