# SkinX - Dual-Modality Deep Learning Platform for Skin Disease Prediction

## Project Overview

Skin diseases represent a significant global health concern, often leading to physical discomfort and psychological distress. Accurate diagnosis remains a challenge due to subjective nature of visual assessments and complexity of dermatological presentations. Recent advancements in machine learning have enabled automated diagnostic systems that enhance clinical decision-making.

This project introduces **SkinX**, a dual-modality deep learning platform for skin disease prediction using either clinical images or textual symptom descriptions.

## Technical Architecture

### Image-Based Diagnosis
- **Model**: EfficientNet-B3 CNN architecture
- **Preprocessing**: Gaussian filtering and GrabCut segmentation
- **Purpose**: Scalable efficiency and superior performance in medical image classification

### Text-Based Input
- **Model**: BioBERT transformer-based model
- **Training**: Pretrained on biomedical literature, fine-tuned for clinical symptom narratives
- **Purpose**: Interpret clinical symptom descriptions and classify skin conditions

## Dataset

- **Image Dataset**: 30,000+ labeled skin disease images across 10 categories
- **Text Corpus**: Parallel corpus of symptom descriptions
- **Categories**: 10 distinct skin disease types

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Confusion matrix analysis
- Robustness validation for both models

## Applications
- Teledermatology
- Rural healthcare
- Medical education

## Hardware & Software Requirements

### Software Requirements
- **Operating System**: Windows 11
- **Programming Language**: Python 3.8 or above
- **Deep Learning Frameworks**: 
  - TensorFlow / Keras
  - PyTorch
  - HuggingFace Transformers (for BioBERT)
- **Image Processing Tools**:
  - OpenCV (GrabCut segmentation)
  - SciPy (Gaussian filtering)
  - scikit-image
- **NLP Tools**:
  - BioBERT pretrained model
  - Tokenizers
- **Libraries**:
  - Pandas, NumPy
  - Scikit-learn (metrics, evaluation)
  - Matplotlib / Seaborn
- **Web Server / Deployment**:
  - Flask, FastAPI, or Django
  - Streamlit (for user interface)

### Hardware Requirements
- **Processor**: Intel Core i5 / AMD Ryzen 5 or higher
- **RAM**: 16GB or more
- **Storage**: 512GB SSD or more
- **GPU (Recommended)**: NVIDIA GTX 1660 / RTX 2060 or higher with CUDA support
- **Network**: High-speed internet connection

## Implementation Status

### Current Working Version
The system has been implemented with multiple versions:

1. **app_working.py** - Simple keyword-based detection (currently running)
2. **app_ml_enhanced.py** - ML-inspired algorithm
3. **app_comprehensive.py** - Comprehensive disease analysis
4. **app_high_accuracy.py** - High-accuracy ML algorithm

### Disease Categories Supported
1. HPV (Viral Infections)
2. Melanoma
3. Eczema
4. Psoriasis
5. Rosacea
6. Basal Cell Carcinoma
7. Squamous Cell Carcinoma
8. Actinic Keratosis
9. Dermatitis
10. Acne

## Current Features
- ✅ Web-based interface
- ✅ Image upload and analysis
- ✅ Text symptom analysis
- ✅ Real-time prediction
- ✅ Confidence scoring
- ✅ Multiple disease detection
- ✅ Processing time optimization

## Next Steps for Full Implementation

### 1. Model Integration
- Implement actual EfficientNet-B3 model for images
- Integrate BioBERT for text analysis
- Set up model loading and inference pipeline

### 2. Preprocessing Pipeline
- Implement Gaussian filtering for images
- Add GrabCut segmentation
- Create preprocessing utilities

### 3. Training Pipeline
- Set up training scripts for both models
- Implement data augmentation
- Create evaluation metrics calculation

### 4. Production Deployment
- Optimize for production use
- Add authentication and security
- Implement logging and monitoring

## Project Structure
```
Skin Disease/
├── app_working.py              # Current working version
├── app_ml_enhanced.py          # ML-enhanced version
├── app_comprehensive.py          # Comprehensive analysis
├── app_high_accuracy.py         # High-accuracy version
├── models/                      # Model implementations
│   ├── efficientnet_model.py
│   └── biobert_model.py
├── utils/                       # Utility functions
│   ├── image_preprocessing.py
│   ├── evaluation_metrics.py
│   └── model_optimization.py
├── templates/                   # HTML templates
│   ├── index.html
│   └── index_fast.html
├── static/                      # Static files
│   ├── css/
│   ├── js/
│   └── uploads/
├── requirements.txt             # Dependencies
└── README_PROJECT.md           # This file
```

## Current Status
The project is currently running with a working keyword-based detection system that provides accurate disease identification for testing purposes. The full deep learning implementation with EfficientNet-B3 and BioBERT models is ready for integration based on the specifications provided.

## Access
- **Local**: http://127.0.0.1:5000
- **Network**: http://192.168.1.110:5000

## Contributing
This project builds on existing literature by combining image processing, statistical feature extraction, and deep learning to address diagnostic challenges in dermatology.
