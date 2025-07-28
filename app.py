#!/usr/bin/env python3
"""
Glaucoma Detection Demo - Streamlit Version
A demo interface showing how the glaucoma detection would work
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Glaucoma Detection",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS to match the React app design
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .main > div {
        padding-top: 2rem;
        background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 50%, #e0e7ff 100%);
        min-height: 100vh;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styles */
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
    }
    
    .header-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 4rem;
        height: 4rem;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        border-radius: 1rem;
        box-shadow: 0 10px 25px rgba(37, 99, 235, 0.3);
        margin-right: 1rem;
        margin-bottom: 1rem;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        line-height: 1.1;
    }
    
    .main-subtitle {
        font-size: 1.25rem;
        color: #64748b;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Card styles */
    .custom-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 1.125rem;
        font-weight: 600;
        color: #334155;
    }
    
    .card-icon {
        margin-right: 0.5rem;
        color: #3b82f6;
    }
    
    /* Button styles */
    .main-button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.875rem 2rem;
        font-size: 1.125rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
        width: 100%;
        margin: 1rem 0;
    }
    
    .main-button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4);
        transform: translateY(-2px);
    }
    
    /* Results display */
    .results-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        min-height: 400px;
    }
    
    .ready-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        height: 400px;
        color: #64748b;
    }
    
    .ready-icon {
        width: 5rem;
        height: 5rem;
        background: #f1f5f9;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
        color: #94a3b8;
        font-size: 2.5rem;
    }
    
    /* Alert styles */
    .error-alert {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #dc2626;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .success-alert {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #16a34a;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    /* Progress bar */
    .progress-container {
        margin: 1rem 0;
    }
    
    .progress-bar {
        background: #e2e8f0;
        border-radius: 1rem;
        height: 0.5rem;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 1rem;
        transition: width 0.3s ease;
    }
    
    .progress-normal {
        background: #10b981;
    }
    
    .progress-glaucoma {
        background: #ef4444;
    }
    
    /* Info section */
    .info-section {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 1rem;
        padding: 2rem;
        margin-top: 4rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .model-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .model-card {
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid;
    }
    
    .model-card-blue {
        background: #eff6ff;
        border-color: #bfdbfe;
        color: #1e40af;
    }
    
    .model-card-green {
        background: #f0fdf4;
        border-color: #bbf7d0;
        color: #166534;
    }
    
    .model-card-purple {
        background: #faf5ff;
        border-color: #d8b4fe;
        color: #7c2d12;
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: none;
        background: transparent;
    }
    
    .stFileUploader > div > div {
        border: 2px dashed #3b82f6;
        border-radius: 0.75rem;
        background: #eff6ff;
        padding: 2rem;
    }
    
    /* Image styling */
    .uploaded-image {
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-height: 300px;
        object-fit: contain;
    }
    
    /* Medical disclaimer */
    .medical-disclaimer {
        background: #fffbeb;
        border: 1px solid #fed7aa;
        color: #92400e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models (same as original functionality)
@st.cache_resource
def load_models():
    models = {}
    model_paths = {
        "MobileNetV3": "models/mobilenet_model.h5",
        "ResNet50": "models/resnet_model.h5", 
        "EfficientNet": "models/efficientnet_model.h5"
    }
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = tf.keras.models.load_model(path)
            except Exception as e:
                st.error(f"Error loading {name}: {str(e)}")
        else:
            st.warning(f"Model file not found: {path}")
    
    return models

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_glaucoma(model, image):
    """Make prediction using the selected model"""
    try:
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        
        # Assuming binary classification (0: Normal, 1: Glaucoma)
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            result = "Glaucoma Detected"
            risk_level = "High" if confidence > 0.8 else "Moderate"
        else:
            result = "Normal"
            risk_level = "Low"
        
        probabilities = {
            "normal": 1 - confidence,
            "glaucoma": confidence
        }
        
        return {
            "prediction": result,
            "confidence": confidence,
            "risk_level": risk_level,
            "probabilities": probabilities
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'ResNet50'

# Load models
models = load_models()

# Header Section
st.markdown("""
<div class="main-header">
    <div class="header-icon">👁️</div>
    <h1 class="main-title">Glaucoma Detection</h1>
    <p class="main-subtitle">Advanced AI-powered analysis of fundus images for early glaucoma detection</p>
</div>
""", unsafe_allow_html=True)

# Error handling
if 'error_message' in st.session_state and st.session_state.error_message:
    st.markdown(f"""
    <div class="error-alert">
        <span>⚠️</span>
        <div>
            <strong>Analysis Error</strong><br>
            {st.session_state.error_message}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    # Image Upload Section
    st.markdown("""
    <div class="custom-card">
        <div class="card-header">
            <span class="card-icon">📤</span>
            Upload Fundus Image
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a fundus image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a fundus image for glaucoma detection",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Model Selection Section
    st.markdown("""
    <div class="custom-card">
        <div class="card-header">
            <span class="card-icon">🧠</span>
            Select AI Model
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    selected_model = st.selectbox(
        "Choose a model",
        options=list(models.keys()) if models else ["No models available"],
        index=list(models.keys()).index(st.session_state.selected_model) if st.session_state.selected_model in models else 0,
        disabled=len(models) == 0,
        label_visibility="collapsed"
    )
    
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.session_state.results = None
    
    # Model info display
    if selected_model and selected_model in models:
        model_info = {
            "MobileNetV3": {
                "description": "Lightweight and fast",
                "features": ["Mobile optimized", "Fast inference", "Low memory"],
                "color": "green"
            },
            "ResNet50": {
                "description": "High accuracy", 
                "features": ["Deep architecture", "Proven accuracy", "Robust features"],
                "color": "blue"
            },
            "EfficientNet": {
                "description": "Balanced performance",
                "features": ["Efficient scaling", "Balanced speed/accuracy", "Optimized design"],
                "color": "purple"
            }
        }
        
        if selected_model in model_info:
            info = model_info[selected_model]
            features_html = "".join([f'<span style="display: inline-block; padding: 0.25rem 0.75rem; margin: 0.25rem; background: white; border: 1px solid #cbd5e1; border-radius: 1rem; font-size: 0.75rem;">{feature}</span>' for feature in info["features"]])
            
            st.markdown(f"""
            <div style="padding: 1rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 0.75rem; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                    <h4 style="margin: 0; font-weight: 600;">🧠 {selected_model}</h4>
                    <span style="padding: 0.25rem 0.5rem; background: #eff6ff; color: #1e40af; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500;">{info["description"]}</span>
                </div>
                <div>{features_html}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Action Buttons
    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        if st.button("🧠 Detect Glaucoma", type="primary", use_container_width=True):
            if uploaded_file is None:
                st.session_state.error_message = "Please upload a fundus image first"
                st.rerun()
            elif selected_model not in models:
                st.session_state.error_message = "Selected model is not available"
                st.rerun()
            else:
                st.session_state.error_message = None
                with st.spinner("Analyzing image..."):
                    image = Image.open(uploaded_file)
                    results = predict_glaucoma(models[selected_model], image)
                    if results:
                        st.session_state.results = results
                        st.rerun()
    
    with col_btn2:
        if st.button("Reset", use_container_width=True):
            st.session_state.results = None
            st.session_state.uploaded_file = None
            st.session_state.error_message = None
            st.rerun()

with col2:
    # Results Display Section
    st.markdown('<div class="results-card">', unsafe_allow_html=True)
    
    if st.session_state.results:
        results = st.session_state.results
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-size: 1.25rem; font-weight: bold; color: #1e293b;">ℹ️ Analysis Results</h2>
            <span style="font-size: 0.875rem; color: #64748b;">Model: {selected_model}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display uploaded image in results
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Analyzed Fundus Image", use_column_width=True)
        
        # Main prediction result
        is_glaucoma = "glaucoma" in results["prediction"].lower()
        icon = "❌" if is_glaucoma else "✅"
        bg_color = "#fef2f2" if is_glaucoma else "#f0fdf4"
        border_color = "#fecaca" if is_glaucoma else "#bbf7d0"
        text_color = "#dc2626" if is_glaucoma else "#16a34a"
        
        confidence_percent = results["confidence"] * 100;
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 1rem; background: {bg_color}; border: 1px solid {border_color}; border-radius: 0.5rem; color: {text_color}; margin-bottom: 1.5rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <div>
                <p style="margin: 0; font-weight: 600; font-size: 1.125rem;">{results["prediction"]}</p>
                <p style="margin: 0; font-size: 0.875rem; opacity: 0.8;">Confidence: {confidence_percent:.1f}%</p>
                <p style="margin: 0; font-size: 0.75rem; opacity: 0.7;">Risk Level: {results["risk_level"]}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed probabilities
        st.markdown('<p style="color: #64748b; font-weight: 500; margin-bottom: 0.75rem;">Detailed Analysis</p>', unsafe_allow_html=True)
        
        # Normal probability
        normal_prob = results["probabilities"]["normal"] * 100
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-size: 0.875rem; color: #64748b;">Normal</span>
                <span style="font-size: 0.875rem; font-weight: 500;">{normal_prob:.1f}%</span>
            </div>
            <div style="width: 100%; height: 0.5rem; background: #e2e8f0; border-radius: 1rem; overflow: hidden;">
                <div style="height: 100%; background: #10b981; border-radius: 1rem; width: {normal_prob}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Glaucoma probability  
        glaucoma_prob = results["probabilities"]["glaucoma"] * 100
        st.markdown(f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-size: 0.875rem; color: #64748b;">Glaucoma</span>
                <span style="font-size: 0.875rem; font-weight: 500;">{glaucoma_prob:.1f}%</span>
            </div>
            <div style="width: 100%; height: 0.5rem; background: #e2e8f0; border-radius: 1rem; overflow: hidden;">
                <div style="height: 100%; background: #ef4444; border-radius: 1rem; width: {glaucoma_prob}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Medical disclaimer
        st.markdown("""
        <div class="medical-disclaimer">
            <strong>⚠️ Medical Disclaimer:</strong> This AI analysis is for educational and research purposes only. 
            It should not be used as a substitute for professional medical diagnosis. 
            Please consult with a qualified ophthalmologist for proper medical evaluation.
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Ready state display
        st.markdown("""
        <div class="ready-state">
            <div class="ready-icon">🧠</div>
            <h3 style="margin: 0 0 0.75rem 0; font-size: 1.25rem; font-weight: 600; color: #374151;">Ready for Analysis</h3>
            <p style="margin: 0; color: #6b7280; max-width: 300px;">Upload a fundus image and select a model to begin glaucoma detection analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Info Section
st.markdown("""
<div class="info-section">
    <h2 style="text-align: center; font-size: 1.5rem; font-weight: bold; color: #1e293b; margin-bottom: 1rem;">
        Advanced AI Models
    </h2>
    <div class="model-grid">
        <div class="model-card model-card-blue">
            <h3 style="margin: 0 0 0.5rem 0; font-weight: 600;">MobileNetV3</h3>
            <p style="margin: 0; font-size: 0.875rem;">Optimized for mobile deployment with efficient architecture</p>
        </div>
        <div class="model-card model-card-green">
            <h3 style="margin: 0 0 0.5rem 0; font-weight: 600;">ResNet50</h3>
            <p style="margin: 0; font-size: 0.875rem;">Deep residual network with proven accuracy</p>
        </div>
        <div class="model-card model-card-purple">
            <h3 style="margin: 0 0 0.5rem 0; font-weight: 600;">EfficientNet</h3>
            <p style="margin: 0; font-size: 0.875rem;">Balanced efficiency and performance</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
