"""
Fast Image Preprocessing for Real-time Performance
Optimized for speed while maintaining quality
"""

import cv2
import numpy as np
from functools import lru_cache
import time
import logging

logger = logging.getLogger(__name__)

class FastImagePreprocessor:
    """
    Optimized image preprocessing for real-time performance
    """
    
    def __init__(self, target_size=(300, 300)):
        self.target_size = target_size
        
    @lru_cache(maxsize=50)
    def _resize_cached(self, image_shape, target_size):
        """Cached resize operation"""
        return target_size
    
    def preprocess_fast(self, image_path, quality='balanced'):
        """
        Fast preprocessing with quality options
        
        Args:
            image_path (str): Path to image
            quality (str): 'fast', 'balanced', 'high'
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        start_time = time.time()
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Quality-based preprocessing
            if quality == 'fast':
                processed = self._preprocess_fast(image)
            elif quality == 'balanced':
                processed = self._preprocess_balanced(image)
            else:  # high
                processed = self._preprocess_high_quality(image)
            
            processing_time = time.time() - start_time
            logger.debug(f"Fast preprocessing ({quality}): {processing_time:.3f}s")
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in fast preprocessing: {e}")
            return None
    
    def _preprocess_fast(self, image):
        """Ultra-fast preprocessing (~10-50ms)"""
        # Just resize and normalize
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    
    def _preprocess_balanced(self, image):
        """Balanced speed/quality (~50-200ms)"""
        # Simple blur for noise reduction
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Resize
        resized = cv2.resize(blurred, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Simple contrast enhancement
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Normalize
        normalized = enhanced.astype(np.float32) / 255.0
        return normalized
    
    def _preprocess_high_quality(self, image):
        """High quality processing (~200-500ms)"""
        # Better noise reduction
        blurred = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Simple segmentation (faster than GrabCut)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply mask
        mask = cv2.dilate(thresh, None, iterations=2)
        segmented = cv2.bitwise_and(blurred, blurred, mask=mask)
        
        # Resize with better interpolation
        resized = cv2.resize(segmented, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        # Advanced contrast enhancement
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Normalize
        normalized = enhanced.astype(np.float32) / 255.0
        return normalized
    
    def batch_preprocess(self, image_paths, quality='balanced'):
        """
        Batch preprocessing for multiple images
        
        Args:
            image_paths (list): List of image paths
            quality (str): Quality level
            
        Returns:
            list: List of preprocessed images
        """
        results = []
        start_time = time.time()
        
        for i, path in enumerate(image_paths):
            processed = self.preprocess_fast(path, quality)
            if processed is not None:
                results.append(processed)
            else:
                logger.warning(f"Failed to process image: {path}")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(image_paths) if image_paths else 0
        
        logger.info(f"Batch preprocessing: {len(results)}/{len(image_paths)} images, "
                   f"avg: {avg_time:.3f}s per image")
        
        return results


# Performance comparison utility
def benchmark_preprocessing():
    """Compare preprocessing speeds"""
    import time
    
    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite('test_image.jpg', test_image)
    
    preprocessor = FastImagePreprocessor()
    
    qualities = ['fast', 'balanced', 'high']
    times = {}
    
    for quality in qualities:
        start = time.time()
        for _ in range(10):  # Run 10 times for average
            result = preprocessor.preprocess_fast('test_image.jpg', quality)
        avg_time = (time.time() - start) / 10
        times[quality] = avg_time
        print(f"{quality.capitalize()}: {avg_time:.3f}s")
    
    return times


if __name__ == "__main__":
    benchmark_preprocessing()
