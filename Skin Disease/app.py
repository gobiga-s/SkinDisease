from flask import Flask, render_template, request, jsonify
import os
import uuid
import logging
import base64
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['LOW_CONFIDENCE_THRESHOLD'] = float(os.getenv('LOW_CONFIDENCE_THRESHOLD', '0.60'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

SKIN_DISEASE_CATEGORIES = [
    'Acne', 'Eczema', 'Psoriasis', 'Rosacea', 'Melanoma',
    'Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Actinic Keratosis',
    'Dermatitis', 'Viral Infections'
]
HIGH_RISK_CATEGORIES = {'Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma'}


class SkinDiseasePredictor:
    def __init__(self):
        self.image_model = None
        self.text_model = None
        self.tokenizer = None
        self.torch = None
        self.cv2 = None
        self.enable_text_model = os.getenv('ENABLE_TEXT_MODEL', '0') == '1'
        self.torch_available = False
        self.cv2_available = self._init_cv2()
        self.models_loaded = False

    @property
    def image_model_loaded(self):
        return self.image_model is not None

    @property
    def text_model_loaded(self):
        return self.text_model is not None and self.tokenizer is not None

    def _init_torch(self):
        if self.torch is not None:
            return True
        try:
            import torch as torch_module
            self.torch = torch_module
            return True
        except Exception as exc:
            logger.warning('Torch unavailable; text model inference disabled: %s', exc)
            self.torch = None
            return False

    def _init_cv2(self):
        try:
            import cv2 as cv2_module
            self.cv2 = cv2_module
            return True
        except Exception as exc:
            logger.warning('OpenCV unavailable; using basic preprocessing fallback: %s', exc)
            self.cv2 = None
            return False

    def load_models(self):
        if self.models_loaded:
            return
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

        if not self.enable_text_model:
            logger.info('Text model loading disabled (set ENABLE_TEXT_MODEL=1 to enable)')
            self.text_model = None
            self.tokenizer = None
            self.torch_available = False
            self.models_loaded = True
            return

        self.torch_available = self._init_torch()
        if not self.torch_available:
            self.text_model = None
            self.tokenizer = None
            self.models_loaded = True
            return

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

        self.models_loaded = True

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            if not self.cv2_available:
                return self.preprocess_image_fallback(image_path)

            image = self.cv2.imread(image_path)
            if image is None:
                return None

            filtered = self.cv2.GaussianBlur(image, (5, 5), 0)
            mask = np.zeros(filtered.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            h, w = filtered.shape[:2]
            if h < 20 or w < 20:
                segmented = self.cv2.resize(filtered, (300, 300))
                return segmented / 255.0

            rect = (10, 10, w - 20, h - 20)
            self.cv2.grabCut(filtered, mask, rect, bgd_model, fgd_model, 5, self.cv2.GC_INIT_WITH_RECT)

            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            segmented = filtered * mask2[:, :, np.newaxis]
            segmented = self.cv2.resize(segmented, (300, 300))
            return segmented / 255.0
        except Exception as exc:
            logger.exception('Error preprocessing image: %s', exc)
            return None


    def preprocess_image_fallback(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = tf.keras.utils.load_img(image_path, target_size=(300, 300))
            arr = tf.keras.utils.img_to_array(image) / 255.0
            return arr
        except Exception as exc:
            logger.exception('Fallback image preprocessing failed: %s', exc)
            return None

    def predict_image(self, image_path: str) -> Tuple[Optional[dict], Optional[str]]:
        self.load_models()

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

    def predict_text(self, symptoms_text: str) -> Tuple[Optional[dict], Optional[str]]:
        self.load_models()

        if not self.text_model_loaded:
            return None, 'Text model is unavailable. Please contact administrator.'

        inputs = self.tokenizer(
            symptoms_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512,
        )

        with self.torch.no_grad():
            outputs = self.text_model(**inputs)
            predictions = self.torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = self.torch.argmax(predictions, dim=-1).item()
            confidence = float(self.torch.max(predictions, dim=-1)[0])

        return {
            'disease': SKIN_DISEASE_CATEGORIES[predicted_class],
            'confidence': confidence,
            'input_text': symptoms_text,
        }, None


predictor = SkinDiseasePredictor()


def allowed_image(file) -> bool:
    if not file or not file.filename:
        return False
    allowed_ext = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    ext = os.path.splitext(file.filename.lower())[1]
    return ext in allowed_ext and (file.mimetype or '').startswith('image/')


def build_safety_metadata(disease: str, confidence: float) -> dict:
    low_confidence = confidence < app.config['LOW_CONFIDENCE_THRESHOLD']
    high_risk = disease in HIGH_RISK_CATEGORIES
    message = None

    if high_risk:
        message = 'Potential high-risk finding detected. Seek dermatologist evaluation as soon as possible.'
    elif low_confidence:
        message = 'Low-confidence estimate. Retake a clearer image or consult a dermatologist for confirmation.'

    return {
        'is_low_confidence': low_confidence,
        'is_high_risk': high_risk,
        'recommendation': message,
        'is_medical_diagnosis': False,
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_image', methods=['POST'])
def predict_image():
    file_path = None
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if not allowed_image(file):
            return jsonify({'error': 'Unsupported image type'}), 400

        unique_name = f'upload_{uuid.uuid4().hex}.jpg'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(file_path)

        result, err = predictor.predict_image(file_path)
        if err:
            return jsonify({'error': err}), 503

        result['safety'] = build_safety_metadata(result['disease'], result['confidence'])

        with open(file_path, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({'success': True, 'result': result, 'image': img_base64})
    except Exception:
        logger.exception('Unhandled error in /predict_image')
        return jsonify({'error': 'Internal server error during image prediction'}), 500
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                logger.warning('Could not remove temporary upload file %s', file_path)


@app.route('/predict_text', methods=['POST'])
def predict_text():
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms text provided'}), 400

        symptoms_text = str(data['symptoms']).strip()
        if not symptoms_text:
            return jsonify({'error': 'Symptoms text cannot be empty'}), 400
        if len(symptoms_text) < 10:
            return jsonify({'error': 'Please provide more symptom details (at least 10 characters).'}), 400

        result, err = predictor.predict_text(symptoms_text)
        if err:
            return jsonify({'error': err}), 503

        result['safety'] = build_safety_metadata(result['disease'], result['confidence'])
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
            'torch_available': predictor.torch_available,
            'text_model_enabled': predictor.enable_text_model,
            'cv2_available': predictor.cv2_available,
            'low_confidence_threshold': app.config['LOW_CONFIDENCE_THRESHOLD'],
        }
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
