"""Image processing utilities"""

import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, Optional

def resize_image(image: Image.Image, max_size: int = 1024, maintain_aspect: bool = True) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    
    if maintain_aspect:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)

def preprocess_for_analysis(image: Image.Image) -> np.ndarray:
    """Preprocess image for analysis"""
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize for consistent analysis
    image = resize_image(image, max_size=512)
    
    # Convert to numpy array
    return np.array(image)

def normalize_image(image_array: np.ndarray) -> np.ndarray:
    """Normalize image array to 0-1 range"""
    return image_array.astype(np.float32) / 255.0

def calculate_image_stats(image: Image.Image) -> dict:
    """Calculate basic image statistics"""
    
    img_array = np.array(image)
    
    stats = {
        'width': image.size[0],
        'height': image.size[1],
        'channels': len(img_array.shape),
        'mean_brightness': np.mean(img_array),
        'std_brightness': np.std(img_array),
        'min_value': np.min(img_array),
        'max_value': np.max(img_array)
    }
    
    return stats

def create_difference_map(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Create visual difference map between two images"""
    
    # Ensure same dimensions
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Calculate absolute difference
    diff = np.abs(img1.astype(float) - img2.astype(float))
    
    # Normalize to 0-255 range
    diff_normalized = (diff / np.max(diff) * 255).astype(np.uint8)
    
    return diff_normalized