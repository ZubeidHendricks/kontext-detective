"""Metrics and scoring utilities"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_similarity_metrics(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Calculate various similarity metrics between two images"""
    
    # Mean Squared Error
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Structural Similarity (simplified version)
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim
    }

def calculate_ai_score(metrics: Dict) -> float:
    """Calculate AI likelihood score from various metrics"""
    
    # This is a simplified scoring function
    # In practice, this would be trained on labeled data
    
    scores = []
    
    # Sensitivity scoring
    if 'sensitivity' in metrics:
        avg_sensitivity = np.mean(list(metrics['sensitivity'].values()))
        sensitivity_score = min(avg_sensitivity / 100, 1.0)
        scores.append(sensitivity_score)
    
    # Consistency scoring
    if 'consistency' in metrics:
        consistency_score = 1.0 - metrics['consistency']
        scores.append(consistency_score)
    
    # Artifact scoring
    if 'artifacts' in metrics:
        artifact_scores = []
        for artifacts in metrics['artifacts'].values():
            if 'artifact_density' in artifacts:
                artifact_scores.append(artifacts['artifact_density'])
        
        if artifact_scores:
            avg_artifact_score = np.mean(artifact_scores)
            scores.append(avg_artifact_score)
    
    # Combine scores
    if scores:
        return np.mean(scores)
    else:
        return 0.5

def evaluate_detector_performance(predictions: List[float], 
                                ground_truth: List[bool], 
                                threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate detector performance against ground truth"""
    
    # Convert predictions to binary using threshold
    binary_predictions = [pred > threshold for pred in predictions]
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, binary_predictions)
    precision = precision_score(ground_truth, binary_predictions)
    recall = recall_score(ground_truth, binary_predictions)
    f1 = f1_score(ground_truth, binary_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': threshold
    }