# SkinX Application Analysis

## Purpose
SkinX is a Flask-based web application designed to **classify skin diseases from either images or symptom text**. It exposes two inference endpoints:
- `/predict_image` for uploaded skin photos
- `/predict_text` for symptom descriptions

The project is presented as an educational/research assistant and explicitly includes a medical disclaimer.

## Core Algorithms and Techniques Used

### 1) Image modality
- **Backbone architecture**: EfficientNet-B3 for multi-class image classification.
- **Image preprocessing in app path**:
  - Gaussian blur for noise reduction.
  - GrabCut segmentation for foreground extraction.
  - Resize to 300×300 and normalize to [0, 1].
- **Training-time augmentation (model module)**:
  - Random horizontal flip, rotation, zoom, contrast.
  - Additional generator-based augmentation: width/height shift, shear, brightness, channel shift.
- **Optimization/training**:
  - Adam optimizer.
  - Categorical cross-entropy loss.
  - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.

### 2) Text modality
- **Backbone architecture**: BioBERT (`dmis-lab/biobert-base-cased-v1.1`) via Hugging Face Transformers.
- **NLP pipeline**:
  - Tokenization with truncation/padding to max 512 tokens.
  - Sequence classification head with 10 labels.
  - Softmax over logits for confidence/probabilities.
- **Training/evaluation**:
  - Hugging Face `Trainer` with weighted precision/recall/F1 + accuracy.

## Functionalities
- Responsive UI with Bootstrap + custom CSS/JS.
- Drag-and-drop image upload + preview.
- Symptom textarea + quick symptom tags.
- Displays predicted class and confidence (image and text modes).
- Health endpoint for basic server/model status.
- Separate model utility modules for training/evaluation.

## Inferred Workflow
1. User opens `/` and chooses image-based or text-based analysis.
2. Frontend submits:
   - image file to `/predict_image` (multipart/form-data), or
   - symptom JSON to `/predict_text`.
3. Backend validates input and runs prediction.
4. Backend returns JSON with class and confidence.
5. Frontend renders prediction cards and top image probabilities.

## Major Issues Identified

### A. Production-readiness and accuracy risks
1. **Image model is instantiated with random/untrained weights in `app.py`** (`weights=None`) and used for inference directly if loading succeeds, which can produce meaningless predictions.
2. **Fallback behavior uses random predictions/confidence** when model loading fails; this can mislead end users.
3. **BioBERT path in app initializes a pretrained checkpoint but not clearly fine-tuned domain weights**; practical diagnostic accuracy may be low.
4. **No calibrated confidence / uncertainty handling** (e.g., temperature scaling, abstention threshold).
5. **No ensemble or triage logic** that routes suspected cancer classes to high-sensitivity handling.

### B. Data and evaluation concerns
1. Claims (e.g., high accuracy, large dataset) are not directly tied to reproducible experiment artifacts in repo.
2. No clear dataset governance (class imbalance handling, skin tone diversity, acquisition domain shift).
3. No visible external validation pipeline (clinic/hospital domain test set).
4. No explicit bias/fairness auditing by demographic or Fitzpatrick skin type.

### C. Security, safety, and compliance gaps
1. App stores uploaded file with fixed name (`temp_image.jpg`) -> race/overwrite risk in concurrent use.
2. Input sanitization and malware/image parsing hardening are minimal.
3. Medical UX safety controls are weak (no red-flag escalation protocol, no strict non-diagnostic messaging in results panel).
4. Logging/error responses may reveal internal exceptions.

### D. Engineering/maintainability issues
1. Repository contains many app variants (`app_*.py`), making canonical runtime path unclear.
2. Dependency import check in `run.py` can mis-detect packages (e.g., `opencv-python` import name mismatch).
3. Duplicated/fragmented logic across multiple app entry files increases drift risk.
4. No clear automated test suite in root workflow to validate endpoints/model behavior.

### E. UI/UX issues
1. Confidence bars imply certainty even when predictions may be random/fallback.
2. Limited explainability (no saliency maps, no rationale text, no differential diagnosis notes).
3. Accessibility is basic; no explicit screen-reader strategy, keyboard flow review, or high-contrast mode.
4. No user guidance for image quality (focus, lighting, lesion framing) before upload.

## High-Impact Improvements

### 1) Improve clinical-model accuracy
- Replace demo inference with **strictly loaded trained checkpoints only**; block predictions if model unavailable.
- Fine-tune EfficientNet and BioBERT on curated, balanced, diverse datasets.
- Add **class imbalance handling** (focal loss/class weighting, oversampling).
- Add **calibration** (temperature scaling) and confidence thresholding.
- Introduce **top-k + abstain option** when confidence is low.
- Add **ensemble models** (e.g., EfficientNet + ConvNeXt) for robust performance.
- Perform external validation and publish confusion matrices by subgroup.

### 2) Strengthen safety and trust
- Explicitly display “not a diagnosis” near every result.
- Add **cancer-risk escalation flow** for melanoma/BCC/SCC predictions (urgent consultation guidance).
- Remove random fallback predictions; return explicit “model unavailable”.
- Add audit logging, PHI-safe storage policy, and retention controls.

### 3) Improve backend robustness
- Use unique filenames/UUIDs and isolated temp directories.
- Add schema validation for text payloads and strict MIME/type checks.
- Add rate limiting, structured logging, and safer exception handling.
- Containerize with reproducible model/version pinning and startup health checks.

### 4) Improve UI/UX
- Add image capture guidance overlays and quality checks (blur/exposure).
- Show top-3 predictions with confidence + “why” hints.
- Add explainability panel (Grad-CAM heatmap for image mode).
- Improve accessibility (ARIA labels, contrast, keyboard-first workflows).
- Add progress state granularity and clearer error messages.

## Suggested Next Milestones
1. **Stabilize architecture**: select one canonical `app.py` and archive/remove variants.
2. **Safety-first inference contract**: no random outputs; strict model availability checks.
3. **Evaluation package**: reproducible scripts + metrics dashboard + subgroup analysis.
4. **UI trust redesign**: uncertainty-aware result component and escalation guidance.
5. **MLOps baseline**: model registry, versioned artifacts, CI endpoint tests.
