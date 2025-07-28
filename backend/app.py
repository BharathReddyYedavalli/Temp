#!/usr/bin/env python3
"""
Glaucoma Detection API Server
FastAPI backend for serving trained PyTorch models
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Glaucoma Detection API",
    description="AI-powered glaucoma detection using fundus images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5194", "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
models_cache = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Model configurations with bias correction
MODEL_CONFIGS = {
    'EfficientNet': {
        'model_path': '../Notebooks/EfficientNet/best_glaucoma_model.pth',
        'model_type': 'efficientnet_b0',
        'labels_swapped': True,  # Based on investigation
        'bias_correction': 0.15,  # Reduce Class 0 confidence by 15%
        'threshold': 0.2  # Very low threshold due to severe bias
    },
    'MobileNetV3': {
        'model_path': '../Notebooks/MobileNetV3-Large/best_glaucoma_model.pth',
        'model_type': 'mobilenet_v3_large',
        'labels_swapped': True,
        'bias_correction': 0.20,  # Reduce Class 0 confidence by 20%
        'threshold': 0.15  # Even lower due to extreme bias
    },
    'ResNet50': {
        'model_path': '../Notebooks/ResNet50/best_glaucoma_model.pth',
        'model_type': 'resnet50',
        'labels_swapped': True,
        'bias_correction': 0.15,  # Reduce Class 0 confidence by 15%
        'threshold': 0.2
    }
}

# Image preprocessing transforms
def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_model(model_type):
    """Create model architecture based on type"""
    if model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
    elif model_type == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, 2)
        )
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.classifier[1].in_features, 2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def load_model(model_name):
    """Load a trained model"""
    if model_name in models_cache:
        return models_cache[model_name]
    
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_path = Path(__file__).parent / config['model_path']
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Create model architecture
        model = create_model(config['model_type'])
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        models_cache[model_name] = model
        logger.info(f"Successfully loaded {model_name}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

def preprocess_image(image_bytes):
    """Preprocess uploaded image"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        transform = get_transforms()
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(device)
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Glaucoma Detection API",
        "status": "healthy",
        "device": str(device),
        "available_models": list(MODEL_CONFIGS.keys())
    }

@app.get("/models")
async def get_models():
    """Get available models"""
    return {
        "models": list(MODEL_CONFIGS.keys()),
        "device": str(device)
    }

@app.post("/predict")
async def predict_glaucoma(
    image: UploadFile = File(...),
    model: str = Form(default="ResNet50")
):
    """
    Predict glaucoma from fundus image with bias correction
    """
    try:
        # Validate model
        if model not in MODEL_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available models: {list(MODEL_CONFIGS.keys())}"
            )
        
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read and preprocess image
        image_bytes = await image.read()
        image_tensor = preprocess_image(image_bytes)
        
        # Load model
        model_instance = load_model(model)
        
        # Make prediction
        with torch.no_grad():
            outputs = model_instance(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get raw probabilities
            prob_class_0 = float(probabilities[0][0])
            prob_class_1 = float(probabilities[0][1])
            
            config = MODEL_CONFIGS.get(model, {})
            
            # Apply bias correction
            bias_correction = config.get('bias_correction', 0.0)
            if bias_correction > 0:
                # Reduce Class 0 probability to combat bias
                prob_class_0_corrected = max(0.1, prob_class_0 - bias_correction)
                prob_class_1_corrected = 1.0 - prob_class_0_corrected
                
                logger.info(f"Bias correction applied: {prob_class_0:.3f} → {prob_class_0_corrected:.3f}")
                
                prob_class_0 = prob_class_0_corrected
                prob_class_1 = prob_class_1_corrected
            
            # Apply label mapping
            if config.get('labels_swapped', False):
                prob_normal = prob_class_1
                prob_glaucoma = prob_class_0
            else:
                prob_normal = prob_class_0
                prob_glaucoma = prob_class_1
            
            # Use custom threshold
            threshold = config.get('threshold', 0.5)
            is_glaucoma = prob_glaucoma > threshold
            
            # Calculate final confidence
            final_confidence = max(prob_glaucoma, prob_normal)
            
            logger.info(f"Final: prob_glaucoma={prob_glaucoma:.3f}, threshold={threshold}, prediction={'Glaucoma' if is_glaucoma else 'Normal'}")
        
        # Prepare response
        result = {
            "prediction": "Glaucoma Detected" if is_glaucoma else "No Glaucoma",
            "confidence": final_confidence,
            "probabilities": {
                "normal": prob_normal,
                "glaucoma": prob_glaucoma
            },
            "model_used": model,
            "threshold_used": threshold,
            "bias_corrected": bias_correction > 0,
            "risk_level": "High" if prob_glaucoma > 0.6 else "Medium" if prob_glaucoma > 0.3 else "Low"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Preload models for faster inference
    logger.info("Preloading models...")
    for model_name in MODEL_CONFIGS.keys():
        try:
            load_model(model_name)
            logger.info(f"✓ {model_name} loaded successfully")
        except Exception as e:
            logger.warning(f"✗ Failed to load {model_name}: {str(e)}")
    
    # Start server
    logger.info("Starting Glaucoma Detection API Server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
