"""
Model Optimization for Faster Inference
Implements quantization, pruning, and model compression
"""

import tensorflow as tf
import torch
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """
    Optimize models for faster inference
    """
    
    def __init__(self):
        self.optimized_models = {}
    
    def optimize_tensorflow_model(self, model_path, output_dir, optimization_level='medium'):
        """
        Optimize TensorFlow model for faster inference
        
        Args:
            model_path (str): Path to saved model
            output_dir (str): Directory to save optimized model
            optimization_level (str): 'light', 'medium', 'aggressive'
        """
        try:
            logger.info(f"Optimizing TensorFlow model: {model_path}")
            
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            if optimization_level == 'light':
                # Basic optimization
                optimized_model = self._light_optimization_tf(model)
            elif optimization_level == 'medium':
                # Medium optimization with quantization
                optimized_model = self._medium_optimization_tf(model)
            else:  # aggressive
                # Aggressive optimization
                optimized_model = self._aggressive_optimization_tf(model)
            
            # Save optimized model
            optimized_model.save(output_dir)
            logger.info(f"Optimized model saved to: {output_dir}")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"Error optimizing TensorFlow model: {e}")
            return None
    
    def _light_optimization_tf(self, model):
        """Light optimization - basic pruning"""
        # Apply pruning
        pruning_params = {
            'pruning_schedule': tf.keras.callbacks.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.1,
                begin_step=1000,
                end_step=10000
            )
        }
        
        # Convert to TensorFlow Lite for faster inference
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        return tflite_model
    
    def _medium_optimization_tf(self, model):
        """Medium optimization - dynamic range quantization"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        return tflite_model
    
    def _aggressive_optimization_tf(self, model):
        """Aggressive optimization - full quantization"""
        # Create representative dataset for quantization
        def representative_data_gen():
            for _ in range(100):
                yield [np.random.rand(1, 300, 300, 3).astype(np.float32)]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        return tflite_model
    
    def optimize_pytorch_model(self, model_path, output_dir, optimization_level='medium'):
        """
        Optimize PyTorch model for faster inference
        
        Args:
            model_path (str): Path to model
            output_dir (str): Directory to save optimized model
            optimization_level (str): 'light', 'medium', 'aggressive'
        """
        try:
            logger.info(f"Optimizing PyTorch model: {model_path}")
            
            # Load model
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            
            if optimization_level == 'light':
                optimized_model = self._light_optimization_pytorch(model)
            elif optimization_level == 'medium':
                optimized_model = self._medium_optimization_pytorch(model)
            else:  # aggressive
                optimized_model = self._aggressive_optimization_pytorch(model)
            
            # Save optimized model
            optimized_model.save_pretrained(output_dir)
            logger.info(f"Optimized model saved to: {output_dir}")
            
            return optimized_model
            
        except Exception as e:
            logger.error(f"Error optimizing PyTorch model: {e}")
            return None
    
    def _light_optimization_pytorch(self, model):
        """Light optimization - basic quantization"""
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    def _medium_optimization_pytorch(self, model):
        """Medium optimization - static quantization"""
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with sample data
        sample_input = torch.randn(1, 10, 768)  # Sample input
        model(sample_input)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        return quantized_model
    
    def _aggressive_optimization_pytorch(self, model):
        """Aggressive optimization - pruning + quantization"""
        # Apply structured pruning
        import torch.nn.utils.prune as prune
        
        # Prune linear layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.2)
                prune.remove(module, 'weight')
        
        # Then quantize
        quantized_model = self._medium_optimization_pytorch(model)
        return quantized_model
    
    def benchmark_model(self, model, input_shape, model_type='tensorflow'):
        """
        Benchmark model performance
        
        Args:
            model: Model to benchmark
            input_shape: Input shape for testing
            model_type: 'tensorflow' or 'pytorch'
            
        Returns:
            dict: Performance metrics
        """
        import time
        
        try:
            # Generate random input
            if model_type == 'tensorflow':
                sample_input = np.random.rand(*input_shape).astype(np.float32)
            else:  # pytorch
                sample_input = torch.randn(*input_shape)
            
            # Warm up
            for _ in range(5):
                if model_type == 'tensorflow':
                    _ = model(sample_input, training=False)
                else:
                    with torch.no_grad():
                        _ = model(sample_input)
            
            # Benchmark
            times = []
            for _ in range(100):
                start_time = time.time()
                
                if model_type == 'tensorflow':
                    _ = model(sample_input, training=False)
                else:
                    with torch.no_grad():
                        _ = model(sample_input)
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            return {
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'min_inference_time': min_time,
                'max_inference_time': max_time,
                'throughput': 1.0 / avg_time,
                'samples_per_second': 100.0 / sum(times)
            }
            
        except Exception as e:
            logger.error(f"Error benchmarking model: {e}")
            return None


# Usage example and utility functions
def create_optimized_models():
    """Create optimized versions of models"""
    optimizer = ModelOptimizer()
    
    # Optimize EfficientNet (if exists)
    efficientnet_path = "models/efficientnet_skin_classifier.h5"
    if Path(efficientnet_path).exists():
        optimizer.optimize_tensorflow_model(
            efficientnet_path,
            "models/efficientnet_optimized",
            optimization_level='medium'
        )
    
    # Optimize BioBERT (if exists)
    biobert_path = "models/biobert_skin_classifier"
    if Path(biobert_path).exists():
        optimizer.optimize_pytorch_model(
            biobert_path,
            "models/biobert_optimized",
            optimization_level='medium'
        )


if __name__ == "__main__":
    create_optimized_models()
