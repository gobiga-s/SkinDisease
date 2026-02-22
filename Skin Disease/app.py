from flask import Flask, render_template, request, jsonify
import os
import uuid
import logging
import base64
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    @property
    def image_model_loaded(self):
        return self.image_model is not None

    @property
    def text_model_loaded(self):
        return self.text_model is not None and self.tokenizer is not None

    def load_models(self):
        """Load trained image model and fine-tuned text model if available."""
        image_model_path = os.getenv('EFFICIENTNET_MODEL_PATH', 'models/efficientnet_skin_classifier.h5')
        text_model_path = os.getenv('BIOMODEL_PATH', 'models/biobert_skin_classifier')

        try:
            if os.path.exists(image_model_path):
                self.image_model = tf.keras.models.load_model(image_model_path)
                logger.info('Loaded image model from %s', image_model_path)
            else:
                logger.warning('Image model not found at %s', image_model_path)
        except Exception as exc:
            logger.exception('Failed to load image model: %s', exc)
            self.image_model = None

        try:
            if os.path.exists(text_model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(text_model_path)
                self.text_model = AutoModelForSequenceClassification.from_pretrained(text_model_path)
                logger.info('Loaded text model from %s', text_model_path)
            else:
                logger.warning('Text model not found at %s', text_model_path)
        except Exception as exc:
            logger.exception('Failed to load text model: %s', exc)
            self.text_model = None
            self.tokenizer = None

    def preprocess_image(self, image_path):
        """Apply Gaussian filtering and GrabCut segmentation."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            filtered = cv2.GaussianBlur(image, (5, 5), 0)

            mask = np.zeros(filtered.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            h, w = filtered.shape[:2]
            rect = (10, 10, max(w - 20, 1), max(h - 20, 1))
            cv2.grabCut(filtered, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            segmented = filtered * mask2[:, :, np.newaxis]
            segmented = cv2.resize(segmented, (300, 300))
            return segmented / 255.0

        except Exception as exc:
            logger.exception('Error preprocessing image: %s', exc)
            return None

    def predict_image(self, image_path):
        """Predict skin disease from image."""
        if not self.image_model_loaded:
            return None, 'Image model is unavailable. Please contact administrator.'

        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return None, 'Image preprocessing failed. Please upload a clearer image.'

        processed_image = np.expand_dims(processed_image, axis=0)
        predictions = self.image_model.predict(processed_image, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return {
            'disease': SKIN_DISEASE_CATEGORIES[predicted_class],
            'confidence': confidence,
            'all_predictions': {
                SKIN_DISEASE_CATEGORIES[i]: float(predictions[0][i])
                for i in range(len(SKIN_DISEASE_CATEGORIES))
            }
        }, None

    def predict_text(self, symptoms_text):
        """Predict skin disease from text symptoms."""
        if not self.text_model_loaded:
            return None, 'Text model is unavailable. Please contact administrator.'

        inputs = self.tokenizer(
            symptoms_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = float(torch.max(predictions, dim=-1)[0])

        return {
            'disease': SKIN_DISEASE_CATEGORIES[predicted_class],
            'confidence': confidence,
            'input_text': symptoms_text,
        }, None


predictor = SkinDiseasePredictor()


def allowed_image(file):
    if not file or not file.filename:
        return False
    allowed_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    ext = os.path.splitext(file.filename.lower())[1]
    return ext in allowed_ext and (file.mimetype or '').startswith('image/')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if not allowed_image(file):
            return jsonify({'error': 'Unsupported image type'}), 400

        unique_name = f"upload_{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(file_path)

        result, err = predictor.predict_image(file_path)
        if err:
            return jsonify({'error': err}), 503

        with open(file_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({'success': True, 'result': result, 'image': img_base64})
    except Exception:
        logger.exception('Unhandled error in /predict_image')
        return jsonify({'error': 'Internal server error during image prediction'}), 500


@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json(silent=True)
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms text provided'}), 400

        symptoms_text = str(data['symptoms']).strip()
        if not symptoms_text:
            return jsonify({'error': 'Symptoms text cannot be empty'}), 400

        result, err = predictor.predict_text(symptoms_text)
        if err:
            return jsonify({'error': err}), 503

        return jsonify({'success': True, 'result': result})
    except Exception:
        logger.exception('Unhandled error in /predict_text')
        return jsonify({'error': 'Internal server error during text prediction'}), 500


@app.route('/health')
def health_check():
    return jsonify(
        {
            'status': 'healthy',
            'image_model_loaded': predictor.image_model_loaded,
            'text_model_loaded': predictor.text_model_loaded,
        }
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
