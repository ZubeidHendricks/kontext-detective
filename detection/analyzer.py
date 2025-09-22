"""Image Analysis and Comparison Functions"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, List
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from scipy import ndimage

class ImageAnalyzer:
    """Analyzes images for AI detection patterns"""
    
    def __init__(self):
        self.metrics = []
    
    def preprocess_image(self, image: Image.Image, target_size: int = 512) -> np.ndarray:
        """Preprocess image for analysis"""
        # Resize while maintaining aspect ratio
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        return np.array(image)
    
    def calculate_mse(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Mean Squared Error between two images"""
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray, img2_gray = img1, img2
        
        return ssim(img1_gray, img2_gray, data_range=255)
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        return psnr(img1, img2, data_range=255)
    
    def detect_artifacts(self, original: np.ndarray, edited: np.ndarray) -> Dict[str, float]:
        """Detect reconstruction artifacts"""
        
        # Calculate difference map
        diff = np.abs(original.astype(float) - edited.astype(float))
        
        # Artifact metrics
        artifacts = {
            "mean_difference": np.mean(diff),
            "max_difference": np.max(diff),
            "std_difference": np.std(diff),
            "artifact_density": np.sum(diff > 10) / diff.size,
            "high_freq_artifacts": self._detect_high_freq_artifacts(diff)
        }
        
        return artifacts
    
    def _detect_high_freq_artifacts(self, diff: np.ndarray) -> float:
        """Detect high-frequency artifacts using edge detection"""
        if len(diff.shape) == 3:
            diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            diff_gray = diff.astype(np.uint8)
        
        # Apply Sobel edge detection
        sobel_x = cv2.Sobel(diff_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(diff_gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return np.mean(edges)
    
    def analyze_texture_consistency(self, images: List[np.ndarray]) -> float:
        """Analyze texture consistency across multiple edits"""
        if len(images) < 2:
            return 1.0
        
        consistencies = []
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                # Calculate texture similarity
                consistency = self.calculate_ssim(images[i], images[j])
                consistencies.append(consistency)
        
        return np.mean(consistencies)
    
    def measure_edit_sensitivity(self, original: np.ndarray, 
                               edited_images: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Measure how sensitively image responds to different prompts"""
        
        sensitivities = {}
        
        for prompt, edited in edited_images.items():
            if edited is not None:
                # Calculate multiple sensitivity metrics
                mse = self.calculate_mse(original, edited)
                ssim_score = self.calculate_ssim(original, edited)
                
                # Sensitivity score (higher = more sensitive to edits)
                sensitivity = mse / (ssim_score + 0.001)  # Avoid division by zero
                sensitivities[prompt] = sensitivity
            else:
                sensitivities[prompt] = 0.0
        
        return sensitivities
    
    def comprehensive_analysis(self, original: Image.Image, 
                             edited_results: Dict[str, Image.Image]) -> Dict[str, any]:
        """Perform comprehensive analysis of all editing results"""
        
        # Preprocess images
        original_np = self.preprocess_image(original)
        edited_np = {}
        
        for prompt, img in edited_results.items():
            if img is not None:
                edited_np[prompt] = self.preprocess_image(img)
        
        # Calculate metrics
        analysis = {
            "basic_metrics": {},
            "artifacts": {},
            "sensitivity": {},
            "consistency": 0.0,
            "ai_likelihood": 0.0
        }
        
        # Basic metrics for each edit
        for prompt, edited in edited_np.items():
            analysis["basic_metrics"][prompt] = {
                "mse": self.calculate_mse(original_np, edited),
                "ssim": self.calculate_ssim(original_np, edited),
                "psnr": self.calculate_psnr(original_np, edited)
            }
            
            analysis["artifacts"][prompt] = self.detect_artifacts(original_np, edited)
        
        # Sensitivity analysis
        analysis["sensitivity"] = self.measure_edit_sensitivity(original_np, edited_np)
        
        # Consistency across edits
        if len(edited_np) > 1:
            analysis["consistency"] = self.analyze_texture_consistency(list(edited_np.values()))
        
        # Calculate AI likelihood score
        analysis["ai_likelihood"] = self._calculate_ai_likelihood(analysis)
        
        return analysis
    
    def _calculate_ai_likelihood(self, analysis: Dict) -> float:
        """Calculate likelihood that image is AI-generated based on analysis"""
        
        scores = []
        
        # High sensitivity to edits suggests AI generation
        if analysis["sensitivity"]:
            avg_sensitivity = np.mean(list(analysis["sensitivity"].values()))
            sensitivity_score = min(avg_sensitivity / 1000, 1.0)  # Normalize
            scores.append(sensitivity_score)
        
        # Low consistency across edits suggests AI generation
        consistency_score = 1.0 - analysis["consistency"]
        scores.append(consistency_score)
        
        # High artifact density suggests AI generation
        if analysis["artifacts"]:
            artifact_scores = []
            for prompt_artifacts in analysis["artifacts"].values():
                artifact_score = prompt_artifacts.get("artifact_density", 0)
                artifact_scores.append(artifact_score)
            
            if artifact_scores:
                avg_artifact_score = np.mean(artifact_scores)
                scores.append(avg_artifact_score)
        
        # Combine scores
        if scores:
            return np.mean(scores)
        else:
            return 0.5  # Neutral score if no data