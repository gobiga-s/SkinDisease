# SkinX: AI-Powered Skin Disease Prediction System

SkinX is a comprehensive dual-modality deep learning platform for accurate skin disease prediction using either clinical images or textual symptom descriptions. Built with EfficientNet-B3 for image analysis and BioBERT for text processing, SkinX offers reliable, scalable, and accessible diagnostic capabilities.

## 🌟 Features

### **Dual-Modality Analysis**
- **Image-Based Diagnosis**: Upload skin disease images for AI-powered analysis using EfficientNet-B3
- **Text-Based Diagnosis**: Describe symptoms for analysis using BioBERT NLP model
- **Advanced Preprocessing**: Gaussian filtering and GrabCut segmentation for enhanced image clarity

### **Comprehensive Disease Coverage**
Supports 10 major skin disease categories:
- Acne
- Eczema
- Psoriasis
- Rosacea
- Melanoma
- Basal Cell Carcinoma
- Squamous Cell Carcinoma
- Actinic Keratosis
- Dermatitis
- Viral Infections

### **Professional Web Interface**
- Modern, responsive design with Bootstrap 5
- Real-time prediction with confidence scores
- Interactive result visualization
- Drag-and-drop image upload
- Symptom tag system for quick input

## 🏗️ Architecture

```
SkinX/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Main web interface
├── static/
│   ├── css/style.css             # Custom styling
│   └── js/main.js                # Frontend JavaScript
├── models/
│   ├── efficientnet_model.py     # EfficientNet-B3 implementation
│   └── biobert_model.py          # BioBERT implementation
├── utils/
│   ├── image_preprocessing.py     # Image preprocessing utilities
│   └── evaluation_metrics.py     # Model evaluation tools
├── static/uploads/                # Uploaded images
└── models/                        # Trained model weights
```

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8 or higher
- pip package manager
- 16GB RAM (recommended)
- NVIDIA GPU with CUDA support (recommended for training)

### **Installation**

1. **Clone or download the project**
   ```bash
   # Extract to your desired location
   cd "Skin Disease"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv skinx_env
   skinx_env\Scripts\activate  # On Windows
   # skinx_env/bin/activate     # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

## 📊 Model Details

### **EfficientNet-B3 Image Model**
- **Architecture**: EfficientNet-B3 with custom classification head
- **Input Size**: 300x300 RGB images
- **Preprocessing**: Gaussian filtering + GrabCut segmentation
- **Data Augmentation**: Rotation, flip, zoom, contrast adjustments
- **Training**: 30,000+ labeled images across 10 categories

### **BioBERT Text Model**
- **Architecture**: BioBERT-base-cased-v1.1 with classification head
- **Input**: Symptom descriptions (max 512 tokens)
- **Fine-tuning**: Domain-specific medical symptom corpus
- **Output**: Disease classification with confidence scores

## 🎯 Usage Guide

### **Image-Based Prediction**
1. Click "Choose Image" or drag-and-drop an image
2. Supported formats: JPG, PNG, GIF (max 16MB)
3. Click "Analyze Image" for prediction
4. View results with confidence scores and top predictions

### **Text-Based Prediction**
1. Describe your symptoms in the text area
2. Use quick symptom tags for common conditions
3. Click "Analyze Symptoms" for prediction
4. Review AI-generated diagnosis with confidence

### **API Usage**

#### **Image Prediction**
```python
import requests

with open('skin_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict_image', files=files)
    result = response.json()
```

#### **Text Prediction**
```python
import requests

data = {'symptoms': 'Red, itchy patches on arms that are dry and flaky'}
response = requests.post('http://localhost:5000/predict_text', json=data)
result = response.json()
```

## 🧪 Model Training

### **Training the Image Model**
```python
from models.efficientnet_model import EfficientNetSkinClassifier

# Initialize classifier
classifier = EfficientNetSkinClassifier(num_classes=10)

# Build model
model = classifier.build_model(pretrained=True)

# Create data generators
train_gen, val_gen, test_gen = classifier.create_data_generators(
    train_dir='data/train',
    val_dir='data/val',
    test_dir='data/test'
)

# Train model
history = classifier.train(train_gen, val_gen, epochs=50)
```

### **Training the Text Model**
```python
from models.biobert_model import BioBERTSkinClassifier

# Initialize classifier
classifier = BioBERTSkinClassifier(num_classes=10)

# Load pretrained model
classifier.load_model()

# Prepare data
texts, labels = classifier.prepare_data('symptom_data.csv')
train_dataset, val_dataset, test_dataset = classifier.create_datasets(texts, labels)

# Fine-tune model
trainer = classifier.train_model(train_dataset, val_dataset)
```

## 📈 Evaluation Metrics

The system provides comprehensive evaluation metrics:

### **Classification Metrics**
- Accuracy, Precision, Recall, F1-Score
- Per-class performance analysis
- Confusion matrix visualization
- ROC curves and AUC scores
- Precision-Recall curves

### **Usage Example**
```python
from utils.evaluation_metrics import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(class_names)

# Generate comprehensive report
report = evaluator.generate_evaluation_report(y_true, y_pred, y_prob)
evaluator.save_evaluation_report(report, 'evaluation_report.json')
```

## 🔧 Configuration

### **Environment Variables**
```bash
# Model paths
EFFICIENTNET_MODEL_PATH=./models/efficientnet_skin_classifier.h5
BIOMODEL_PATH=./models/biobert_skin_classifier

# Server settings
FLASK_ENV=development
FLASK_DEBUG=True
HOST=0.0.0.0
PORT=5000
```

### **Custom Settings**
Edit `app.py` to modify:
- Upload file size limits
- Model paths and parameters
- Disease categories
- Confidence thresholds

## 🎨 Customization

### **Adding New Disease Categories**
1. Update `SKIN_DISEASE_CATEGORIES` in `app.py`
2. Retrain models with new categories
3. Update UI labels in `templates/index.html`
4. Modify class names in model files

### **Custom Preprocessing**
Edit `utils/image_preprocessing.py` to modify:
- Gaussian filter parameters
- GrabCut segmentation settings
- Contrast enhancement methods
- Additional image augmentations

## 🐛 Troubleshooting

### **Common Issues**

1. **Model Loading Errors**
   - Ensure all dependencies are installed
   - Check model file paths
   - Verify GPU availability (if using CUDA)

2. **Memory Issues**
   - Reduce batch size in training
   - Use smaller image dimensions
   - Close unnecessary applications

3. **Upload Failures**
   - Check file size limits (16MB max)
   - Verify supported image formats
   - Ensure upload directory permissions

4. **Prediction Errors**
   - Validate input image format
   - Check symptom text length
   - Verify model loading status

### **Performance Optimization**
- Use GPU acceleration for training
- Implement model quantization for deployment
- Optimize image preprocessing pipeline
- Cache frequent predictions

## 📚 Dataset

The system is designed to work with the [Skin Diseases Image Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset) from Kaggle.

### **Dataset Structure**
```
dataset/
├── train/
│   ├── Acne/
│   ├── Eczema/
│   └── ... (other categories)
├── validation/
└── test/
```

### **Data Requirements**
- Images: JPG/PNG format, minimum 224x224 pixels
- Text: CSV with 'symptoms' and 'disease' columns
- Labels: Exact matches with disease categories

## 🔒 Security & Privacy

- **Local Processing**: All predictions processed locally
- **No Data Storage**: Images not permanently stored
- **HIPAA Compliance**: Suitable for medical use cases
- **Secure Upload**: File type and size validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with medical device regulations if used in clinical settings.

## 🙏 Acknowledgments

- [EfficientNet](https://arxiv.org/abs/1905.11946) - Google Research
- [BioBERT](https://arxiv.org/abs/1901.08746) - DMIS Lab
- [Kaggle Skin Diseases Dataset](https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset)
- Bootstrap, Font Awesome, and other open-source libraries

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the project repository
- Check the troubleshooting section
- Review the documentation and code comments

---

**⚠️ Medical Disclaimer**: This system is for educational and research purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns.
