"""Main Detection Pipeline"""

import time
from typing import Dict, Any, Callable, Optional
from PIL import Image
import numpy as np

from .kontext_client import KontextClient
from .analyzer import ImageAnalyzer

class KontextDetector:
    """Main AI detection class using FLUX.1 Kontext [dev]"""
    
    def __init__(self, api_key: str):
        self.client = KontextClient(api_key)
        self.analyzer = ImageAnalyzer()
        
        # Default edit prompts for testing
        self.default_prompts = [
            "enhance image quality",
            "make colors more vibrant", 
            "improve lighting",
            "sharpen details",
            "reduce noise",
            "increase contrast",
            "brighten image",
            "make more realistic"
        ]
    
    def test_connection(self) -> bool:
        """Test API connection"""
        return self.client.test_connection()
    
    def detect(self, 
               image: Image.Image,
               sensitivity: float = 0.7,
               num_prompts: int = 5,
               custom_prompts: Optional[list] = None,
               progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Main detection function"""
        
        start_time = time.time()
        
        # Select prompts to use
        prompts = custom_prompts or self.default_prompts[:num_prompts]
        
        if progress_callback:
            progress_callback(0.1)
        
        # Step 1: Apply Kontext edits
        edited_results = self.client.batch_edit(image, prompts)
        
        if progress_callback:
            progress_callback(0.6)
        
        # Step 2: Analyze results
        analysis = self.analyzer.comprehensive_analysis(image, edited_results)
        
        if progress_callback:
            progress_callback(0.9)
        
        # Step 3: Generate final results
        processing_time = time.time() - start_time
        
        results = {
            "ai_likelihood": analysis["ai_likelihood"],
            "confidence": self._calculate_confidence(analysis),
            "processing_time": processing_time,
            "sensitivity": analysis["sensitivity"],
            "artifacts": analysis["artifacts"],
            "consistency": analysis["consistency"],
            "basic_metrics": analysis["basic_metrics"],
            "detection_method": "FLUX.1 Kontext [dev] Reconstruction Analysis",
            "prompts_used": prompts,
            "successful_edits": len([r for r in edited_results.values() if r is not None])
        }
        
        if progress_callback:
            progress_callback(1.0)
        
        return results
    
    def _calculate_confidence(self, analysis: Dict) -> float:
        """Calculate confidence score for the detection"""
        
        # Base confidence on consistency of signals
        signals = []
        
        # Sensitivity signal strength
        if analysis["sensitivity"]:
            sensitivity_values = list(analysis["sensitivity"].values())
            sensitivity_std = np.std(sensitivity_values)
            # Lower std = more consistent = higher confidence
            signals.append(1.0 - min(sensitivity_std / 100, 1.0))
        
        # Artifact consistency
        if analysis["artifacts"]:
            artifact_densities = []
            for artifacts in analysis["artifacts"].values():
                artifact_densities.append(artifacts.get("artifact_density", 0))
            
            if artifact_densities:
                artifact_std = np.std(artifact_densities)
                signals.append(1.0 - min(artifact_std, 1.0))
        
        # Overall consistency score
        signals.append(analysis["consistency"])
        
        # Combine signals
        if signals:
            confidence = np.mean(signals)
            # Ensure minimum confidence
            return max(confidence, 0.5)
        else:
            return 0.5
    
    def batch_detect(self, images: list, **kwargs) -> list:
        """Detect multiple images"""
        results = []
        
        for i, image in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}")
            result = self.detect(image, **kwargs)
            results.append(result)
        
        return results