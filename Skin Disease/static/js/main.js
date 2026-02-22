// SkinX Main JavaScript File

// Disease categories information
const diseaseCategories = [
    {
        name: 'Acne',
        description: 'Common skin condition causing pimples, blackheads, and whiteheads',
        icon: 'fa-circle'
    },
    {
        name: 'Eczema',
        description: 'Inflammatory condition causing dry, itchy, and red skin',
        icon: 'fa-hand-paper'
    },
    {
        name: 'Psoriasis',
        description: 'Autoimmune condition causing scaly, red patches on the skin',
        icon: 'fa-layer-group'
    },
    {
        name: 'Rosacea',
        description: 'Chronic condition causing redness and visible blood vessels in the face',
        icon: 'fa-face-red'
    },
    {
        name: 'Melanoma',
        description: 'Serious type of skin cancer that develops in melanocytes',
        icon: 'fa-exclamation-triangle'
    },
    {
        name: 'Basal Cell Carcinoma',
        description: 'Most common type of skin cancer, usually appears as a flesh-colored bump',
        icon: 'fa-circle'
    },
    {
        name: 'Squamous Cell Carcinoma',
        description: 'Common type of skin cancer that can appear as scaly red patches',
        icon: 'fa-square'
    },
    {
        name: 'Actinic Keratosis',
        description: 'Rough, scaly patch on the skin caused by sun exposure',
        icon: 'fa-sun'
    },
    {
        name: 'Dermatitis',
        description: 'General term for skin inflammation causing rash and itching',
        icon: 'fa-allergies'
    },
    {
        name: 'Viral Infections',
        description: 'Skin conditions caused by viruses like warts and herpes',
        icon: 'fa-virus'
    }
];

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadArea();
    initializeSymptomForm();
    populateDiseaseCategories();
});

// Initialize upload area
function initializeUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const analyzeBtn = document.getElementById('analyzeImageBtn');

    // Click to upload
    uploadArea.addEventListener('click', function() {
        imageInput.click();
    });

    // File input change
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageUpload(files[0]);
        }
    });
}

// Handle image upload
function handleImageUpload(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showAlert('Please upload an image file', 'warning');
        return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size must be less than 16MB', 'warning');
        return;
    }

    // Preview image
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewImg = document.getElementById('previewImg');
        const imagePreview = document.getElementById('imagePreview');
        const analyzeBtn = document.getElementById('analyzeImageBtn');
        
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        analyzeBtn.style.display = 'block';
        
        // Hide previous results
        document.getElementById('imageResult').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Analyze image
async function analyzeImage() {
    const imageInput = document.getElementById('imageInput');
    const file = imageInput.files[0];
    
    if (!file) {
        showAlert('Please select an image first', 'warning');
        return;
    }

    showLoading();

    try {
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch('/predict_image', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            displayImageResult(result.result, result.image);
        } else {
            showAlert(result.error || 'Analysis failed', 'danger');
        }
    } catch (error) {
        showAlert('Error analyzing image: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

// Display image analysis result
function displayImageResult(result, imageData) {
    const resultDiv = document.getElementById('imageResult');
    const contentDiv = document.getElementById('imagePredictionContent');
    
    // Create HTML for results
    let html = `
        <div class="mb-3">
            <h5 class="text-success">
                <i class="fas fa-check-circle me-2"></i>
                ${result.disease}
            </h5>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
            </div>
            <p class="mb-2">Confidence: <strong>${(result.confidence * 100).toFixed(1)}%</strong></p>
        </div>
    `;

    // Add all predictions if available
    if (result.all_predictions) {
        html += '<h6 class="mt-3 mb-2">All Predictions:</h6>';
        const sortedPredictions = Object.entries(result.all_predictions)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3); // Top 3 predictions
        
        sortedPredictions.forEach(([disease, confidence]) => {
            const confidencePercent = (confidence * 100).toFixed(1);
            html += `
                <div class="disease-prediction">
                    <span class="disease-name">${disease}</span>
                    <span class="confidence-score">${confidencePercent}%</span>
                </div>
            `;
        });
    }

    contentDiv.innerHTML = html;
    resultDiv.style.display = 'block';
    resultDiv.classList.add('fade-in');
}

// Initialize symptom form
function initializeSymptomForm() {
    const form = document.getElementById('symptomForm');
    
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        await analyzeSymptoms();
    });
}

// Analyze symptoms
async function analyzeSymptoms() {
    const symptomsTextarea = document.getElementById('symptoms');
    const symptoms = symptomsTextarea.value.trim();
    
    if (!symptoms) {
        showAlert('Please describe your symptoms', 'warning');
        return;
    }

    showLoading();

    try {
        const response = await fetch('/predict_text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symptoms: symptoms })
        });

        const result = await response.json();

        if (result.success) {
            displayTextResult(result.result);
        } else {
            showAlert(result.error || 'Analysis failed', 'danger');
        }
    } catch (error) {
        showAlert('Error analyzing symptoms: ' + error.message, 'danger');
    } finally {
        hideLoading();
    }
}

// Display text analysis result
function displayTextResult(result) {
    const resultDiv = document.getElementById('textResult');
    const contentDiv = document.getElementById('textPredictionContent');
    
    const html = `
        <div class="mb-3">
            <h5 class="text-info">
                <i class="fas fa-brain me-2"></i>
                ${result.disease}
            </h5>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
            </div>
            <p class="mb-2">Confidence: <strong>${(result.confidence * 100).toFixed(1)}%</strong></p>
        </div>
        <div class="mt-3">
            <h6>Analyzed Symptoms:</h6>
            <p class="text-muted">${result.input_text}</p>
        </div>
    `;

    contentDiv.innerHTML = html;
    resultDiv.style.display = 'block';
    resultDiv.classList.add('fade-in');
}

// Add symptom tag to textarea
function addSymptom(symptom) {
    const textarea = document.getElementById('symptoms');
    const currentValue = textarea.value.trim();
    
    if (currentValue) {
        textarea.value = currentValue + ', ' + symptom;
    } else {
        textarea.value = symptom;
    }
    
    // Focus back to textarea
    textarea.focus();
}

// Populate disease categories
function populateDiseaseCategories() {
    const container = document.getElementById('diseaseCategories');
    
    diseaseCategories.forEach(category => {
        const col = document.createElement('div');
        col.className = 'col-md-6 col-lg-4';
        
        col.innerHTML = `
            <div class="disease-category">
                <h6>
                    <i class="fas ${category.icon} me-2"></i>
                    ${category.name}
                </h6>
                <p>${category.description}</p>
            </div>
        `;
        
        container.appendChild(col);
    });
}

// Show loading modal
function showLoading() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

// Hide loading modal
function hideLoading() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) {
        modal.hide();
    }
}

// Show alert message
function showAlert(message, type) {
    // Create alert container if it doesn't exist
    let alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alertContainer';
        alertContainer.className = 'position-fixed top-0 start-50 translate-middle-x mt-3';
        alertContainer.style.zIndex = '9999';
        document.body.appendChild(alertContainer);
    }

    // Create alert element
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show alert-custom`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    alertContainer.appendChild(alert);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 5000);
}

// Utility function to format confidence
function formatConfidence(confidence) {
    return (confidence * 100).toFixed(1) + '%';
}

// Utility function to get confidence color
function getConfidenceColor(confidence) {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'danger';
}

// Initialize tooltips if needed
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Add smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + U to focus on upload
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        document.getElementById('imageInput').click();
    }
    
    // Ctrl/Cmd + S to focus on symptoms
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        document.getElementById('symptoms').focus();
    }
});

// Add page visibility change handler
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        // Page is hidden, pause any ongoing operations
        console.log('Page hidden');
    } else {
        // Page is visible again
        console.log('Page visible');
    }
});

// Error handling for unhandled promise rejections
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showAlert('An unexpected error occurred', 'danger');
});

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    initializeTooltips();
    console.log('SkinX application initialized');
});
