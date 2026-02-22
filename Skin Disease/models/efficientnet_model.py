"""
EfficientNet-B3 Model Implementation for Skin Disease Classification
Implements the EfficientNet-B3 architecture for skin disease image classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientNetSkinClassifier:
    """
    EfficientNet-B3 based classifier for skin disease prediction
    """
    
    def __init__(self, num_classes=10, input_shape=(300, 300, 3), learning_rate=0.001):
        """
        Initialize the EfficientNet classifier
        
        Args:
            num_classes (int): Number of skin disease categories
            input_shape (tuple): Input image shape
            learning_rate (float): Learning rate for optimization
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
        # Skin disease categories
        self.class_names = [
            'Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Melanoma',
            'Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Actinic Keratosis',
            'Dermatitis', 'Viral Infections'
        ]
        
    def build_model(self, pretrained=True, fine_tune_at=100):
        """
        Build the EfficientNet-B3 model for skin disease classification
        
        Args:
            pretrained (bool): Whether to use pretrained weights
            fine_tune_at (int): Layer number from which to start fine-tuning
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        try:
            # Load base EfficientNet-B3 model
            if pretrained:
                base_model = EfficientNetB3(
                    weights='imagenet',
                    include_top=False,
                    input_shape=self.input_shape
                )
            else:
                base_model = EfficientNetB3(
                    weights=None,
                    include_top=False,
                    input_shape=self.input_shape
                )
            
            # Freeze the base model layers initially
            base_model.trainable = False
            
            # Build the custom classification head
            inputs = layers.Input(shape=self.input_shape)
            
            # Data augmentation layers
            x = layers.RandomFlip("horizontal")(inputs)
            x = layers.RandomRotation(0.1)(x)
            x = layers.RandomZoom(0.1)(x)
            x = layers.RandomContrast(0.1)(x)
            
            # Preprocessing for EfficientNet
            x = layers.Rescaling(1./255)(x)
            
            # Pass through base model
            x = base_model(x, training=False)
            
            # Global average pooling
            x = layers.GlobalAveragePooling2D()(x)
            
            # Batch normalization
            x = layers.BatchNormalization()(x)
            
            # Dense layers with dropout
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.BatchNormalization()(x)
            
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.BatchNormalization()(x)
            
            # Output layer
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
            # Create the model
            model = models.Model(inputs=inputs, outputs=outputs)
            
            # Compile the model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.model = model
            
            logger.info(f"Model built successfully. Total parameters: {model.count_params():,}")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def fine_tune_model(self, fine_tune_at=100):
        """
        Fine-tune the model by unfreezing top layers
        
        Args:
            fine_tune_at (int): Layer number from which to start fine-tuning
        """
        try:
            if self.model is None:
                raise ValueError("Model must be built before fine-tuning")
            
            # Unfreeze the base model from the specified layer
            base_model = self.model.layers[4]  # EfficientNet base model layer
            
            # Freeze all layers before fine_tune_at
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            
            # Unfreeze layers from fine_tune_at onwards
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
            
            # Re-compile with a lower learning rate
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate / 10),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            logger.info(f"Model fine-tuned from layer {fine_tune_at}")
            
        except Exception as e:
            logger.error(f"Error fine-tuning model: {str(e)}")
            raise
    
    def create_data_generators(self, train_dir, val_dir, test_dir=None, batch_size=32):
        """
        Create data generators for training, validation, and testing
        
        Args:
            train_dir (str): Directory containing training images
            val_dir (str): Directory containing validation images
            test_dir (str): Directory containing test images (optional)
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        try:
            # Training data generator with augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=False,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest',
                brightness_range=[0.8, 1.2],
                channel_shift_range=20
            )
            
            # Validation data generator (only rescaling)
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            # Test data generator (only rescaling)
            test_datagen = ImageDataGenerator(rescale=1./255)
            
            # Create generators
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True,
                classes=self.class_names
            )
            
            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False,
                classes=self.class_names
            )
            
            test_generator = None
            if test_dir and os.path.exists(test_dir):
                test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=self.input_shape[:2],
                    batch_size=batch_size,
                    class_mode='categorical',
                    shuffle=False,
                    classes=self.class_names
                )
            
            logger.info(f"Data generators created successfully")
            logger.info(f"Training samples: {train_generator.samples}")
            logger.info(f"Validation samples: {val_generator.samples}")
            if test_generator:
                logger.info(f"Test samples: {test_generator.samples}")
            
            return train_generator, val_generator, test_generator
            
        except Exception as e:
            logger.error(f"Error creating data generators: {str(e)}")
            raise
    
    def train(self, train_generator, val_generator, epochs=50, model_save_path='best_model.h5'):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of training epochs
            model_save_path (str): Path to save the best model
            
        Returns:
            History: Training history
        """
        try:
            # Define callbacks
            callbacks = [
                ModelCheckpoint(
                    model_save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train the model
            logger.info("Starting model training...")
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            self.history = history
            logger.info("Model training completed successfully")
            
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate(self, test_generator):
        """
        Evaluate the model on test data
        
        Args:
            test_generator: Test data generator
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model must be trained before evaluation")
            
            # Evaluate the model
            results = self.model.evaluate(test_generator, verbose=1)
            
            # Create metrics dictionary
            metrics = {
                'loss': results[0],
                'accuracy': results[1],
                'precision': results[2],
                'recall': results[3]
            }
            
            logger.info(f"Evaluation results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def predict(self, image, return_probabilities=True):
        """
        Make predictions on a single image or batch of images
        
        Args:
            image: Input image(s) - can be single image or batch
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            numpy.ndarray: Predictions
        """
        try:
            if self.model is None:
                raise ValueError("Model must be loaded before prediction")
            
            # Ensure input is in correct format
            if len(image.shape) == 3:
                # Single image - add batch dimension
                image = np.expand_dims(image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(image)
            
            if return_probabilities:
                return predictions
            else:
                return np.argmax(predictions, axis=1)
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def load_model(self, model_path):
        """
        Load a trained model from file
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from: {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def save_model(self, model_path):
        """
        Save the trained model to file
        
        Args:
            model_path (str): Path to save the model
        """
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            self.model.save(model_path)
            logger.info(f"Model saved successfully to: {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def get_model_summary(self):
        """
        Get model architecture summary
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "No model built yet"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        try:
            if self.history is None:
                logger.warning("No training history available")
                return
            
            import matplotlib.pyplot as plt
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot training & validation accuracy
            ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)
            
            # Plot training & validation loss
            ax2.plot(self.history.history['loss'], label='Training Loss')
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")


# Utility functions
def create_model_from_config(config):
    """
    Create a model from configuration dictionary
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        EfficientNetSkinClassifier: Configured model instance
    """
    classifier = EfficientNetSkinClassifier(
        num_classes=config.get('num_classes', 10),
        input_shape=config.get('input_shape', (300, 300, 3)),
        learning_rate=config.get('learning_rate', 0.001)
    )
    
    classifier.build_model(
        pretrained=config.get('pretrained', True),
        fine_tune_at=config.get('fine_tune_at', 100)
    )
    
    return classifier


if __name__ == "__main__":
    # Example usage
    classifier = EfficientNetSkinClassifier()
    
    # Build the model
    model = classifier.build_model(pretrained=True)
    print(classifier.get_model_summary())
    
    # Example training (would need actual data directories)
    # train_gen, val_gen, test_gen = classifier.create_data_generators(
    #     train_dir="data/train",
    #     val_dir="data/val",
    #     test_dir="data/test"
    # )
    # 
    # history = classifier.train(train_gen, val_gen, epochs=50)
    # classifier.plot_training_history("training_history.png")
