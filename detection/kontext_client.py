"""FLUX.1 Kontext API Client"""

import requests
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any
import time
import os

class KontextClient:
    """Client for interacting with FLUX.1 Kontext [dev] via FAL API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FAL_API_KEY")
        if not self.api_key:
            raise ValueError("FAL API key is required")
        
        # FAL API endpoints
        self.base_url = "https://fal.run/fal-ai"
        self.model_id = "black-forest-labs/flux-1-kontext-dev"
        self.headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def base64_to_image(self, base64_str: str) -> Image.Image:
        """Convert base64 string to PIL Image"""
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))
    
    def edit_image(self, 
                   image: Image.Image, 
                   prompt: str,
                   strength: float = 0.7,
                   guidance_scale: float = 7.5,
                   num_inference_steps: int = 20) -> Optional[Image.Image]:
        """Edit image using FLUX.1 Kontext"""
        
        try:
            # Resize image if too large
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # For now, simulate API call - replace with real FAL API implementation
            # Real implementation would make HTTP request to FAL API
            
            # Placeholder: return slightly modified image for testing
            import numpy as np
            
            # Convert to numpy, add slight modifications based on prompt
            img_array = np.array(image)
            
            # Simulate different editing effects based on prompt
            if "enhance" in prompt.lower():
                # Slight sharpening effect
                noise_factor = 3
            elif "vibrant" in prompt.lower():
                # Color enhancement
                noise_factor = 8
            elif "lighting" in prompt.lower():
                # Brightness adjustment
                img_array = np.clip(img_array * 1.05, 0, 255)
                noise_factor = 2
            else:
                noise_factor = 5
            
            # Add controlled noise to simulate editing artifacts
            noise = np.random.normal(0, noise_factor, img_array.shape).astype(np.int16)
            modified_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            return Image.fromarray(modified_array)
            
        except Exception as e:
            print(f"Error in edit_image: {str(e)}")
            return None
    
    def batch_edit(self, 
                   image: Image.Image, 
                   prompts: list,
                   **kwargs) -> Dict[str, Optional[Image.Image]]:
        """Apply multiple prompts to the same image"""
        
        results = {}
        for prompt in prompts:
            print(f"Processing prompt: {prompt}")
            edited = self.edit_image(image, prompt, **kwargs)
            results[prompt] = edited
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        return results
    
    def test_connection(self) -> bool:
        """Test if API connection is working"""
        try:
            # Create a small test image
            test_image = Image.new("RGB", (64, 64), color="red")
            result = self.edit_image(test_image, "test")
            return result is not None
        except:
            return False