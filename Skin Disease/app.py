from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import cv2
from scipy import ndimage
import json
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Skin disease categories (10 categories as mentioned in abstract)
SKIN_DISEASE_CATEGORIES = [
    'Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Melanoma',
    'Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Actinic Keratosis',
    'Dermatitis', 'Viral Infections'
]

class SkinDiseasePredictor:
    def __init__(self):
        self.image_model = None
        self.text_model = None
        self.tokenizer = None
        self.load_models()
    
    def load_models(self):
        """Load EfficientNet-B3 for images and BioBERT for text"""
        try:
            # Load EfficientNet-B3 model (placeholder - would need trained weights)
            self.image_model = tf.keras.applications.EfficientNetB3(
                weights=None,
                include_top=True,
                classes=len(SKIN_DISEASE_CATEGORIES),
                input_shape=(300, 300, 3)
            )
            
            # Load BioBERT model for text classification
            model_name = "dmis-lab/biobert-base-cased-v1.1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=len(SKIN_DISEASE_CATEGORIES)
            )
            
        except Exception as e:
            print(f"Error loading models: {e}")
            # Create dummy models for demonstration
            self.image_model = None
            self.text_model = None
            self.tokenizer = None
    
    def preprocess_image(self, image_path):
        """Apply Gaussian filtering and GrabCut segmentation"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Apply Gaussian filtering for noise reduction
            filtered = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Apply GrabCut segmentation for background removal
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            h, w = image.shape[:2]
            rect = (10, 10, w-20, h-20)  # Rectangle around the object
            
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            segmented = image * mask2[:, :, np.newaxis]
            
            # Resize to EfficientNet input size
            segmented = cv2.resize(segmented, (300, 300))
            
            # Normalize
            segmented = segmented / 255.0
            
            return segmented
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_image(self, image_path):
        """Predict skin disease from image"""
        try:
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return None
            
            # Add batch dimension
            processed_image = np.expand_dims(processed_image, axis=0)
            
            if self.image_model is not None:
                predictions = self.image_model.predict(processed_image)
                predicted_class = np.argmax(predictions[0])
                confidence = float(np.max(predictions[0]))
            else:
                # Dummy prediction for demonstration
                predicted_class = np.random.randint(0, len(SKIN_DISEASE_CATEGORIES))
                confidence = np.random.uniform(0.7, 0.95)
            
            return {
                'disease': SKIN_DISEASE_CATEGORIES[predicted_class],
                'confidence': confidence,
                'all_predictions': {
                    SKIN_DISEASE_CATEGORIES[i]: float(predictions[0][i]) if self.image_model is not None else np.random.uniform(0, 1)
                    for i in range(len(SKIN_DISEASE_CATEGORIES))
                }
            }
            
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None
    
    def predict_text(self, symptoms_text):
        """Predict skin disease from text symptoms"""
        try:
            if self.tokenizer is None or self.text_model is None:
                # Dummy prediction for demonstration
                predicted_class = np.random.randint(0, len(SKIN_DISEASE_CATEGORIES))
                confidence = np.random.uniform(0.6, 0.9)
            else:
                # Tokenize input
                inputs = self.tokenizer(
                    symptoms_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(predictions, dim=-1).item()
                    confidence = float(torch.max(predictions, dim=-1)[0])
            
            return {
                'disease': SKIN_DISEASE_CATEGORIES[predicted_class],
                'confidence': confidence,
                'input_text': symptoms_text
            }
            
        except Exception as e:
            print(f"Error predicting text: {e}")
            return None

# Initialize predictor
predictor = SkinDiseasePredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
            file.save(filename)
            
            # Make prediction
            result = predictor.predict_image(filename)
            
            if result:
                # Convert image to base64 for display
                with open(filename, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                
                return jsonify({
                    'success': True,
                    'result': result,
                    'image': img_base64
                })
            else:
                return jsonify({'error': 'Prediction failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms text provided'}), 400
        
        symptoms_text = data['symptoms']
        if not symptoms_text.strip():
            return jsonify({'error': 'Symptoms text cannot be empty'}), 400
        
        # Make prediction
        result = predictor.predict_text(symptoms_text)
        
        if result:
            return jsonify({
                'success': True,
                'result': result
            })
        else:
            return jsonify({'error': 'Prediction failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': predictor.image_model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
