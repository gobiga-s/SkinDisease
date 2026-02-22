"""
Image Preprocessing Utilities for SkinX Skin Disease Prediction
Implements Gaussian filtering and GrabCut segmentation for enhanced image analysis
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import exposure, filters
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Advanced image preprocessing pipeline for skin disease images
    Implements Gaussian filtering, GrabCut segmentation, and enhancement techniques
    """
    
    def __init__(self):
        self.target_size = (300, 300)  # EfficientNet-B3 input size
        
    def preprocess_image(self, image_path, enhance_contrast=True, apply_noise_reduction=True):
        """
        Complete preprocessing pipeline for skin disease images
        
        Args:
            image_path (str): Path to the input image
            enhance_contrast (bool): Whether to apply contrast enhancement
            apply_noise_reduction (bool): Whether to apply noise reduction
            
        Returns:
            numpy.ndarray: Preprocessed image ready for model input
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            logger.info(f"Processing image: {image_path}, shape: {image.shape}")
            
            # Convert to RGB (OpenCV uses BGR by default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing steps
            processed = self._apply_preprocessing_pipeline(
                image_rgb, 
                enhance_contrast=enhance_contrast,
                apply_noise_reduction=apply_noise_reduction
            )
            
            # Resize to target size
            processed = cv2.resize(processed, self.target_size)
            
            # Normalize to [0, 1]
            processed = processed.astype(np.float32) / 255.0
            
            logger.info(f"Preprocessing completed. Final shape: {processed.shape}")
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def _apply_preprocessing_pipeline(self, image, enhance_contrast=True, apply_noise_reduction=True):
        """
        Apply the complete preprocessing pipeline
        
        Args:
            image (numpy.ndarray): Input RGB image
            enhance_contrast (bool): Whether to apply contrast enhancement
            apply_noise_reduction (bool): Whether to apply noise reduction
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        processed = image.copy()
        
        # Step 1: Noise reduction using Gaussian filtering
        if apply_noise_reduction:
            processed = self._apply_gaussian_filter(processed)
        
        # Step 2: GrabCut segmentation for background removal
        processed = self._apply_grabcut_segmentation(processed)
        
        # Step 3: Contrast enhancement
        if enhance_contrast:
            processed = self._enhance_contrast(processed)
        
        # Step 4: Additional enhancements
        processed = self._apply_additional_enhancements(processed)
        
        return processed
    
    def _apply_gaussian_filter(self, image, kernel_size=(5, 5), sigma=0):
        """
        Apply Gaussian filtering for noise reduction
        
        Args:
            image (numpy.ndarray): Input image
            kernel_size (tuple): Size of the Gaussian kernel
            sigma (float): Standard deviation for Gaussian kernel
            
        Returns:
            numpy.ndarray: Filtered image
        """
        try:
            filtered = cv2.GaussianBlur(image, kernel_size, sigma)
            logger.debug(f"Applied Gaussian filter with kernel size: {kernel_size}")
            return filtered
        except Exception as e:
            logger.error(f"Error applying Gaussian filter: {str(e)}")
            return image
    
    def _apply_grabcut_segmentation(self, image):
        """
        Apply GrabCut segmentation for background removal
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Segmented image with background removed
        """
        try:
            # Convert to uint8 if necessary
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            h, w = image.shape[:2]
            
            # Initialize mask
            mask = np.zeros((h, w), np.uint8)
            
            # Define rectangle around the object (leave some margin)
            margin = 10
            rect = (margin, margin, w - 2*margin, h - 2*margin)
            
            # Initialize background and foreground models
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut algorithm
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create binary mask where foreground is white and background is black
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Apply mask to original image
            segmented = image * mask2[:, :, np.newaxis]
            
            logger.debug("Applied GrabCut segmentation")
            return segmented
            
        except Exception as e:
            logger.error(f"Error applying GrabCut segmentation: {str(e)}")
            return image
    
    def _enhance_contrast(self, image):
        """
        Enhance image contrast using adaptive histogram equalization
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Contrast-enhanced image
        """
        try:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            logger.debug("Applied contrast enhancement")
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing contrast: {str(e)}")
            return image
    
    def _apply_additional_enhancements(self, image):
        """
        Apply additional image enhancements
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        try:
            # Apply unsharp masking for sharpening
            enhanced = self._unsharp_mask(image)
            
            # Apply gamma correction for better brightness
            enhanced = self._gamma_correction(enhanced, gamma=1.2)
            
            logger.debug("Applied additional enhancements")
            return enhanced
            
        except Exception as e:
            logger.error(f"Error applying additional enhancements: {str(e)}")
            return image
    
    def _unsharp_mask(self, image, sigma=1.0, strength=1.5):
        """
        Apply unsharp masking for image sharpening
        
        Args:
            image (numpy.ndarray): Input image
            sigma (float): Standard deviation for Gaussian blur
            strength (float): Strength of sharpening
            
        Returns:
            numpy.ndarray: Sharpened image
        """
        try:
            # Create Gaussian blurred version
            blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            
            # Apply unsharp masking
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            
            # Clip values to valid range
            sharpened = np.clip(sharpened, 0, 255)
            
            return sharpened.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error applying unsharp mask: {str(e)}")
            return image
    
    def _gamma_correction(self, image, gamma=1.0):
        """
        Apply gamma correction for brightness adjustment
        
        Args:
            image (numpy.ndarray): Input image
            gamma (float): Gamma value
            
        Returns:
            numpy.ndarray: Gamma-corrected image
        """
        try:
            # Build lookup table
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            
            # Apply gamma correction using the lookup table
            corrected = cv2.LUT(image, table)
            
            return corrected
            
        except Exception as e:
            logger.error(f"Error applying gamma correction: {str(e)}")
            return image
    
    def batch_preprocess(self, image_paths, output_dir=None):
        """
        Preprocess a batch of images
        
        Args:
            image_paths (list): List of image file paths
            output_dir (str): Directory to save preprocessed images (optional)
            
        Returns:
            list: List of preprocessed images
        """
        preprocessed_images = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Preprocess image
            processed = self.preprocess_image(image_path)
            
            if processed is not None:
                preprocessed_images.append(processed)
                
                # Save if output directory is specified
                if output_dir:
                    import os
                    filename = os.path.basename(image_path)
                    output_path = os.path.join(output_dir, f"preprocessed_{filename}")
                    cv2.imwrite(output_path, (processed * 255).astype(np.uint8))
            else:
                logger.warning(f"Failed to preprocess image: {image_path}")
        
        logger.info(f"Successfully preprocessed {len(preprocessed_images)}/{len(image_paths)} images")
        return preprocessed_images
    
    def visualize_preprocessing_steps(self, image_path, save_path=None):
        """
        Visualize each step of the preprocessing pipeline
        
        Args:
            image_path (str): Path to input image
            save_path (str): Path to save visualization (optional)
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Read original image
            original = cv2.imread(image_path)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing steps individually
            gaussian_filtered = self._apply_gaussian_filter(original_rgb)
            segmented = self._apply_grabcut_segmentation(gaussian_filtered)
            contrast_enhanced = self._enhance_contrast(segmented)
            final = self._apply_additional_enhancements(contrast_enhanced)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            images = [
                (original_rgb, 'Original Image'),
                (gaussian_filtered, 'Gaussian Filtered'),
                (segmented, 'GrabCut Segmented'),
                (contrast_enhanced, 'Contrast Enhanced'),
                (final, 'Final Result'),
                (None, '')
            ]
            
            for i, (img, title) in enumerate(images):
                if img is not None:
                    axes[i].imshow(img)
                    axes[i].set_title(title)
                    axes[i].axis('off')
                else:
                    axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None


# Utility functions for common operations
def load_and_preprocess_image(image_path, target_size=(300, 300)):
    """
    Utility function to load and preprocess a single image
    
    Args:
        image_path (str): Path to the image
        target_size (tuple): Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    preprocessor = ImagePreprocessor()
    preprocessor.target_size = target_size
    return preprocessor.preprocess_image(image_path)


def create_preprocessing_pipeline(config=None):
    """
    Create a preprocessing pipeline with custom configuration
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        ImagePreprocessor: Configured preprocessor instance
    """
    preprocessor = ImagePreprocessor()
    
    if config:
        if 'target_size' in config:
            preprocessor.target_size = config['target_size']
        # Add more configuration options as needed
    
    return preprocessor


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor()
    
    # Test preprocessing on a sample image
    sample_image_path = "sample_skin_image.jpg"  # Replace with actual path
    
    if os.path.exists(sample_image_path):
        # Preprocess single image
        processed = preprocessor.preprocess_image(sample_image_path)
        print(f"Preprocessed image shape: {processed.shape}")
        
        # Create visualization
        fig = preprocessor.visualize_preprocessing_steps(
            sample_image_path, 
            save_path="preprocessing_visualization.png"
        )
        
        # Batch preprocessing example
        image_paths = [sample_image_path]  # Add more paths as needed
        batch_processed = preprocessor.batch_preprocess(image_paths)
        print(f"Batch processed {len(batch_processed)} images")
    else:
        print(f"Sample image not found: {sample_image_path}")
