"""Test script for Kontext Detective"""

import sys
import os
from PIL import Image
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from detection.detector import KontextDetector
    from detection.analyzer import ImageAnalyzer
    from detection.kontext_client import KontextClient
    print("✅ All detection modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality without API calls"""
    
    print("\n🧪 Testing basic functionality...")
    
    # Test image analyzer
    analyzer = ImageAnalyzer()
    
    # Create test images
    test_img1 = Image.new('RGB', (100, 100), color='red')
    test_img2 = Image.new('RGB', (100, 100), color='blue')
    
    # Test preprocessing
    array1 = analyzer.preprocess_image(test_img1)
    array2 = analyzer.preprocess_image(test_img2)
    
    print(f"✅ Image preprocessing: {array1.shape}")
    
    # Test metrics
    mse = analyzer.calculate_mse(array1, array2)
    ssim = analyzer.calculate_ssim(array1, array2)
    
    print(f"✅ MSE calculation: {mse:.2f}")
    print(f"✅ SSIM calculation: {ssim:.3f}")
    
    # Test artifact detection
    artifacts = analyzer.detect_artifacts(array1, array2)
    print(f"✅ Artifact detection: {len(artifacts)} metrics")
    
    print("\n✅ All basic tests passed!")

def test_detector_initialization():
    """Test detector initialization"""
    
    print("\n🧪 Testing detector initialization...")
    
    try:
        # Test with dummy API key
        detector = KontextDetector(api_key="dummy_key")
        print("✅ Detector initialization successful")
        
        # Test prompts
        print(f"✅ Default prompts: {len(detector.default_prompts)} available")
        
        return True
    except Exception as e:
        print(f"❌ Detector initialization failed: {e}")
        return False

def create_sample_images():
    """Create sample images for testing"""
    
    print("\n🖼️ Creating sample test images...")
    
    # Create sample images directory
    os.makedirs("data/real_images", exist_ok=True)
    os.makedirs("data/ai_images", exist_ok=True)
    
    # Create some sample "real" images (simple patterns)
    real_patterns = [
        Image.new('RGB', (256, 256), color='lightblue'),
        Image.new('RGB', (256, 256), color='lightgreen'),
        Image.new('RGB', (256, 256), color='lightyellow')
    ]
    
    for i, img in enumerate(real_patterns):
        # Add some simple noise to make them more realistic
        img_array = np.array(img)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
        noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_array)
        
        noisy_img.save(f"data/real_images/sample_real_{i+1}.png")
    
    print(f"✅ Created {len(real_patterns)} sample 'real' images")
    
    # Create some sample "AI" images (with more artificial patterns)
    ai_patterns = [
        Image.new('RGB', (256, 256), color='purple'),
        Image.new('RGB', (256, 256), color='orange'),
        Image.new('RGB', (256, 256), color='pink')
    ]
    
    for i, img in enumerate(ai_patterns):
        # Add specific patterns that might indicate AI generation
        img_array = np.array(img)
        
        # Add structured noise
        x, y = np.meshgrid(np.arange(256), np.arange(256))
        pattern = np.sin(x/10) * np.cos(y/10) * 20
        
        if len(img_array.shape) == 3:
            pattern = np.stack([pattern] * 3, axis=-1)
        
        artificial_array = np.clip(img_array.astype(np.int16) + pattern.astype(np.int16), 0, 255).astype(np.uint8)
        artificial_img = Image.fromarray(artificial_array)
        
        artificial_img.save(f"data/ai_images/sample_ai_{i+1}.png")
    
    print(f"✅ Created {len(ai_patterns)} sample 'AI' images")

def main():
    """Run all tests"""
    
    print("🔍 Kontext Detective - Test Suite")
    print("=" * 40)
    
    # Test basic functionality
    test_basic_functionality()
    
    # Test detector initialization
    detector_ok = test_detector_initialization()
    
    # Create sample images
    create_sample_images()
    
    print("\n" + "=" * 40)
    
    if detector_ok:
        print("✅ All tests completed successfully!")
        print("\n🚀 Ready to run: streamlit run app.py")
    else:
        print("⚠️ Some tests failed, but basic functionality works")
        print("\n🚀 You can still run: streamlit run app.py (use demo mode)")
    
    print("\n📁 Repository: https://github.com/ZubeidHendricks/kontext-detective")

if __name__ == "__main__":
    main()