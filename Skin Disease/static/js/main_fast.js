// Ultra-Fast SkinX JavaScript - Optimized for Instant Response

// Disease categories (minimal data for speed)
const diseaseCategories = [
    {name: 'Acne', description: 'Pimples, blackheads, whiteheads', icon: 'fa-circle'},
    {name: 'Eczema', description: 'Dry, itchy, inflamed skin', icon: 'fa-hand-paper'},
    {name: 'Psoriasis', description: 'Scaly, silvery patches', icon: 'fa-layer-group'},
    {name: 'Rosacea', description: 'Facial redness, visible vessels', icon: 'fa-face-red'},
    {name: 'Melanoma', description: 'Dark irregular mole, serious', icon: 'fa-exclamation-triangle'},
    {name: 'Basal Cell Carcinoma', description: 'Pearly bump, non-healing sore', icon: 'fa-circle'},
    {name: 'Squamous Cell Carcinoma', description: 'Scaly red patch, crusty', icon: 'fa-square'},
    {name: 'Actinic Keratosis', description: 'Rough patches, sun damage', icon: 'fa-sun'},
    {name: 'Dermatitis', description: 'General skin inflammation', icon: 'fa-allergies'},
    {name: 'Viral Infections', description: 'Warts, herpes, blisters', icon: 'fa-virus'}
];

// Initialize everything instantly
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadArea();
    initializeSymptomForm();
    populateDiseaseCategories();
});

// Upload area (optimized)
function initializeUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const analyzeBtn = document.getElementById('analyzeImageBtn');

    uploadArea.addEventListener('click', () => imageInput.click());
    
    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleImageUpload(file);
    });

    // Drag and drop (simplified)
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleImageUpload(files[0]);
    });
}

// Handle image upload (fast)
function handleImageUpload(file) {
    if (!file.type.startsWith('image/')) {
        showAlert('Please upload an image file', 'warning');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showAlert('File size must be less than 16MB', 'warning');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImg').src = e.target.result;
        document.getElementById('imagePreview').style.display = 'block';
        document.getElementById('analyzeImageBtn').style.display = 'block';
        document.getElementById('imageResult').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Analyze image (instant)
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

// Display image result (optimized)
function displayImageResult(result, imageData) {
    const resultDiv = document.getElementById('imageResult');
    const contentDiv = document.getElementById('imagePredictionContent');
    
    const confidencePercent = (result.confidence * 100).toFixed(1);
    
    const html = `
        <div class="mb-3">
            <h5 class="text-success">
                <i class="fas fa-check-circle me-2"></i>
                ${result.disease}
            </h5>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
            </div>
            <p class="mb-2">Confidence: <strong>${confidencePercent}%</strong></p>
            <p class="mb-2">Processing Time: <strong>${result.processing_time_ms || 'N/A'}ms</strong></p>
        </div>
    `;

    contentDiv.innerHTML = html;
    resultDiv.style.display = 'block';
    resultDiv.classList.add('fade-in');
}

// Initialize symptom form (fast)
function initializeSymptomForm() {
    const form = document.getElementById('symptomForm');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await analyzeSymptoms();
    });
}

// Analyze symptoms (instant)
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

// Display text result (optimized)
function displayTextResult(result) {
    const resultDiv = document.getElementById('textResult');
    const contentDiv = document.getElementById('textPredictionContent');
    
    const confidencePercent = (result.confidence * 100).toFixed(1);
    
    const html = `
        <div class="mb-3">
            <h5 class="text-info">
                <i class="fas fa-brain me-2"></i>
                ${result.disease}
            </h5>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
            </div>
            <p class="mb-2">Confidence: <strong>${confidencePercent}%</strong></p>
            <p class="mb-2">Processing Time: <strong>${result.processing_time_ms || 'N/A'}ms</strong></p>
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

// Add symptom tag (instant)
function addSymptom(symptom) {
    const textarea = document.getElementById('symptoms');
    const currentValue = textarea.value.trim();
    
    if (currentValue) {
        textarea.value = currentValue + ', ' + symptom;
    } else {
        textarea.value = symptom;
    }
    
    textarea.focus();
}

// Populate disease categories (fast)
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

// Show loading (minimal delay)
function showLoading() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
    
    // Auto-hide after 500ms max (since backend is instant)
    setTimeout(() => {
        hideLoading();
    }, 500);
}

// Hide loading (fast)
function hideLoading() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) {
        modal.hide();
    }
}

// Show alert (instant)
function showAlert(message, type) {
    let alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alertContainer';
        alertContainer.className = 'position-fixed top-0 start-50 translate-middle-x mt-3';
        alertContainer.style.zIndex = '9999';
        document.body.appendChild(alertContainer);
    }

    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show alert-custom`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    alertContainer.appendChild(alert);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.remove();
        }
    }, 3000);
}

// Keyboard shortcuts (instant)
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        document.getElementById('imageInput').click();
    }
    
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        document.getElementById('symptoms').focus();
    }
});

// Error handling (instant)
window.addEventListener('unhandledrejection', function(event) {
    console.error('Unhandled promise rejection:', event.reason);
    showAlert('An unexpected error occurred', 'danger');
});

console.log('SkinX Ultra-Fast JavaScript loaded - Instant responses ready!');
