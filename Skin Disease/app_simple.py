"""
Simplified SkinX Application - Works without heavy ML dependencies
Demonstrates the web interface with mock predictions
"""

from flask import Flask, render_template, request, jsonify
import os
import random
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Skin disease categories
SKIN_DISEASE_CATEGORIES = [
    'Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Melanoma',
    'Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Actinic Keratosis',
    'Dermatitis', 'Viral Infections'
]

class MockSkinDiseasePredictor:
    """
    Mock predictor for demonstration purposes
    Returns realistic predictions without requiring ML models
    """
    
    def __init__(self):
        self.disease_descriptions = {
            'Acne': 'Common skin condition causing pimples, blackheads, and whiteheads',
            'Eczema': 'Inflammatory condition causing dry, itchy, and red skin',
            'Psoriasis': 'Autoimmune condition causing scaly, red patches on the skin',
            'Rosacea': 'Chronic condition causing redness and visible blood vessels in the face',
            'Melanoma': 'Serious type of skin cancer that develops in melanocytes',
            'Basal Cell Carcinoma': 'Most common type of skin cancer, usually appears as a flesh-colored bump',
            'Squamous Cell Carcinoma': 'Common type of skin cancer that can appear as scaly red patches',
            'Actinic Keratosis': 'Rough, scaly patch on the skin caused by sun exposure',
            'Dermatitis': 'General term for skin inflammation causing rash and itching',
            'Viral Infections': 'Skin conditions caused by viruses like warts and herpes'
        }
    
    def predict_image(self, image_path):
        """Mock image prediction"""
        # Generate realistic predictions
        predictions = {}
        remaining_confidence = 1.0
        
        for i, disease in enumerate(SKIN_DISEASE_CATEGORIES):
            if i == len(SKIN_DISEASE_CATEGORIES) - 1:
                # Last disease gets remaining confidence
                confidence = remaining_confidence
            else:
                # Generate random confidence with higher values for first few
                if i < 3:
                    confidence = random.uniform(0.1, 0.4)
                else:
                    confidence = random.uniform(0.01, 0.1)
                remaining_confidence -= confidence
            
            predictions[disease] = confidence
        
        # Sort by confidence and normalize
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        total = sum(conf for _, conf in sorted_predictions[:5])
        normalized_predictions = {disease: conf/total for disease, conf in sorted_predictions[:5]}
        
        predicted_disease = sorted_predictions[0][0]
        confidence = normalized_predictions[predicted_disease]
        
        return {
            'disease': predicted_disease,
            'confidence': confidence,
            'description': self.disease_descriptions[predicted_disease],
            'all_predictions': normalized_predictions
        }
    
    def predict_text(self, symptoms_text):
        """Mock text prediction"""
        # Simple keyword-based prediction for demo
        text_lower = symptoms_text.lower()
        
        # Keyword mapping to diseases
        keyword_mapping = {
            'acne': 'Acne',
            'pimple': 'Acne',
            'itchy': 'Eczema',
            'dry': 'Eczema',
            'eczema': 'Eczema',
            'scaly': 'Psoriasis',
            'psoriasis': 'Psoriasis',
            'red': 'Rosacea',
            'rosacea': 'Rosacea',
            'mole': 'Melanoma',
            'melanoma': 'Melanoma',
            'bump': 'Basal Cell Carcinoma',
            'cancer': 'Melanoma',
            'rash': 'Dermatitis',
            'dermatitis': 'Dermatitis',
            'viral': 'Viral Infections',
            'wart': 'Viral Infections'
        }
        
        # Find matching disease
        predicted_disease = 'Eczema'  # default
        for keyword, disease in keyword_mapping.items():
            if keyword in text_lower:
                predicted_disease = disease
                break
        
        # Generate confidence based on text length and keyword matches
        base_confidence = 0.6
        keyword_matches = sum(1 for keyword in keyword_mapping.keys() if keyword in text_lower)
        confidence = min(0.95, base_confidence + (keyword_matches * 0.1) + (len(symptoms_text) / 1000))
        
        # Generate other predictions
        other_diseases = [d for d in SKIN_DISEASE_CATEGORIES if d != predicted_disease]
        other_predictions = {}
        remaining_confidence = 1.0 - confidence
        
        for disease in other_diseases[:4]:
            other_predictions[disease] = remaining_confidence / 4
        
        all_predictions = {predicted_disease: confidence}
        all_predictions.update(other_predictions)
        
        return {
            'disease': predicted_disease,
            'confidence': confidence,
            'description': self.disease_descriptions[predicted_disease],
            'input_text': symptoms_text,
            'all_predictions': all_predictions
        }

# Initialize mock predictor
predictor = MockSkinDiseasePredictor()

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
            # Save the file temporarily
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_image.jpg')
            file.save(filename)
            
            # Make prediction
            result = predictor.predict_image(filename)
            
            # Convert image to base64 for display
            import base64
            with open(filename, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'result': result,
                'image': img_base64,
                'timestamp': datetime.now().isoformat()
            })
    
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
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'mode': 'demo',
        'models_loaded': True,
        'message': 'Running with mock predictions for demonstration'
    })

@app.route('/api/diseases')
def get_diseases():
    """Get list of all supported diseases with descriptions"""
    return jsonify({
        'diseases': [
            {
                'name': disease,
                'description': predictor.disease_descriptions.get(disease, 'Skin disease condition')
            }
            for disease in SKIN_DISEASE_CATEGORIES
        ]
    })

@app.route('/api/stats')
def get_stats():
    """Get demo statistics"""
    return jsonify({
        'total_predictions': random.randint(1000, 5000),
        'accuracy': f"{random.uniform(85, 95):.1f}%",
        'categories': len(SKIN_DISEASE_CATEGORIES),
        'models': ['EfficientNet-B3 (Demo)', 'BioBERT (Demo)']
    })

if __name__ == '__main__':
    print("🩺 SkinX Demo Server Starting...")
    print("🌐 This is a demonstration version with mock predictions")
    print("📊 For full ML functionality, install all dependencies from requirements.txt")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
