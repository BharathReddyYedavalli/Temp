# Glaucoma Detection Backend

This is the FastAPI backend server for the Glaucoma Detection application.

## Setup Instructions

### 1. Install Python Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Model Files

Ensure the following model files exist:
- `../Notebooks/EfficientNet/best_glaucoma_model.pth`
- `../Notebooks/MobileNetV3-Large/best_glaucoma_model.pth`
- `../Notebooks/ResNet50/best_glaucoma_model.pth`

### 3. Run the Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### GET /
Health check endpoint

### GET /models
Get list of available models

### POST /predict
Upload an image and get glaucoma prediction

**Parameters:**
- `image`: Image file (JPG, PNG)
- `model`: Model name (EfficientNet, MobileNetV3, ResNet50)

**Response:**
```json
{
  "prediction": "Glaucoma Detected" | "No Glaucoma",
  "confidence": 0.95,
  "probabilities": {
    "normal": 0.05,
    "glaucoma": 0.95
  },
  "model_used": "ResNet50",
  "risk_level": "High" | "Medium" | "Low"
}
```

## Model Label Configuration

**Important**: All models were trained with swapped labels:
- Class 0 = Glaucoma 
- Class 1 = Normal

The API automatically handles this label mapping to provide correct predictions.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended for loading all models
