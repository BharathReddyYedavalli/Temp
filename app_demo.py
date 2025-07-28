#!/usr/bin/env python3
"""
Glaucoma Detection Demo - Streamlit Version
A demo interface showing how the glaucoma detection would work
"""

import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Page configuration
st.set_page_config(
    page_title="🔬 GlaucoAI - Glaucoma Detection Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_demo_heatmap(image_pil):
    """Create a demo heatmap overlay for demonstration purposes"""
    # Resize image to standard size
    img_resized = image_pil.resize((224, 224))
    img_np = np.array(img_resized.convert("RGB"))
    
    # Create a demo heatmap (simulating Grad-CAM)
    # Focus on center region (where optic disc usually is)
    h, w = 224, 224
    heatmap = np.zeros((h, w))
    
    # Create circular attention pattern in center
    center_x, center_y = w // 2, h // 2
    for i in range(h):
        for j in range(w):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < 50:  # Attention radius
                heatmap[i, j] = max(0, 1 - dist / 50)
    
    # Add some random attention spots
    np.random.seed(42)  # For consistent demo
    for _ in range(3):
        rand_x = np.random.randint(30, w-30)
        rand_y = np.random.randint(30, h-30)
        for i in range(max(0, rand_y-15), min(h, rand_y+15)):
            for j in range(max(0, rand_x-15), min(w, rand_x+15)):
                dist = np.sqrt((i - rand_y)**2 + (j - rand_x)**2)
                if dist < 15:
                    heatmap[i, j] = max(heatmap[i, j], 0.3 * (1 - dist / 15))
    
    # Convert to heatmap colors
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Overlay on original image
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay

def demo_prediction(image_pil, model_name):
    """Generate demo prediction results"""
    # Simulate different results based on model selection
    results = {
        'EfficientNet': {
            'pred_class': 0,  # Normal
            'confidence': 0.78,
            'label': 'Normal',
            'risk_level': 'Low Risk'
        },
        'MobileNetV3': {
            'pred_class': 1,  # Glaucoma
            'confidence': 0.65,
            'label': 'Glaucoma Detected',
            'risk_level': 'High Risk'
        },
        'ResNet50': {
            'pred_class': 0,  # Normal
            'confidence': 0.84,
            'label': 'Normal',
            'risk_level': 'Low Risk'
        }
    }
    
    result = results.get(model_name, results['ResNet50'])
    
    # Create heatmap
    overlay = create_demo_heatmap(image_pil)
    
    # Generate explanation
    if result['pred_class'] == 1:
        explanation = f"""**🔴 GLAUCOMA DETECTED**

Confidence: {result['confidence']*100:.1f}%

The model identified areas of concern in the optic nerve region. The red/yellow areas in the heatmap show regions that contributed to this diagnosis.

⚠️ **Please consult an ophthalmologist immediately for professional evaluation.**

*Note: This is a DEMO. Real models would provide actual medical analysis.*"""
    else:
        explanation = f"""**✅ NORMAL RETINA**

Confidence: {result['confidence']*100:.1f}%

No significant signs of glaucoma detected. The highlighted areas show regions the model analyzed.

💡 **Note: This is for screening purposes only. Regular eye exams are still recommended.**

*Note: This is a DEMO. Real models would provide actual medical analysis.*"""
    
    return overlay, explanation, result['label'], result['confidence'], result['risk_level']

# Streamlit UI
def main():
    # Header
    st.title("🔬 GlaucoAI - Advanced Glaucoma Detection")
    st.markdown("""
    **DEMO VERSION** - Upload a **fundus/retinal image** to see how AI glaucoma detection works. The system shows a **heatmap visualization** highlighting areas the model focuses on.
    
    ⚠️ **Medical Disclaimer**: This tool is for educational and screening purposes only. Always consult with a qualified ophthalmologist for proper medical diagnosis.
    
    🚨 **This is a demonstration only** - Results are simulated for educational purposes.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 Model Selection")
        st.info("🚨 **DEMO MODE** - Using simulated AI models")
        
        available_models = ['EfficientNet', 'MobileNetV3', 'ResNet50']
        
        for model_name in available_models:
            st.success(f"✅ {model_name}: Demo Available")
        
        selected_model = st.selectbox(
            "Choose AI Model:",
            available_models,
            index=2,
            help="Select which model to demo (different models may give different results)"
        )
        
        st.markdown("### 📋 Instructions:")
        st.markdown("""
        1. **Upload** any retinal/eye image
        2. **Select** an AI model to demo
        3. **View** the simulated heatmap and results
        
        **Heatmap Colors:**
        - 🔴 **Red/Orange**: High attention areas
        - 🟡 **Yellow**: Moderate attention areas  
        - 🔵 **Blue**: Low attention areas
        """)
        
        st.markdown("### 🔗 GitHub Repository")
        st.markdown("[View Source Code](https://github.com/BharathReddyYedavalli/AI-Glaucoma-Detection)")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose any retinal/eye image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload any fundus image to see the demo in action"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.header("🎯 Demo Results")
        
        if uploaded_file is not None:
            # Show processing message
            with st.spinner(f'Running {selected_model} demo...'):
                # Run demo prediction
                overlay, explanation, label, confidence, risk_level = demo_prediction(image, selected_model)
                
                # Display results
                st.image(overlay, caption="Grad-CAM Analysis (Demo)", use_column_width=True)
                
                # Show prediction results
                if "GLAUCOMA DETECTED" in explanation:
                    st.error(f"🔴 {label}")
                else:
                    st.success(f"✅ {label}")
                
                # Metrics
                col_conf, col_risk = st.columns(2)
                with col_conf:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                with col_risk:
                    st.metric("Risk Level", risk_level)
                
                # Detailed explanation
                st.markdown("### 📊 Demo Analysis")
                st.markdown(explanation)
                
        else:
            st.info("👆 Please upload any retinal image to see the demo")
            
            # Show example of what real system would look like
            st.markdown("### 🔍 About the Real System")
            st.markdown("""
            The actual GlaucoAI system includes:
            
            **🧠 Three Trained AI Models:**
            - EfficientNet-B0 (Balanced performance)
            - MobileNetV3-Large (Fast inference)  
            - ResNet50 (High accuracy)
            
            **🎯 Advanced Features:**
            - Real Grad-CAM visualization
            - Trained on medical datasets
            - Professional-grade accuracy
            - Multiple model ensemble
            
            **📊 Real Medical Analysis:**
            - Optic disc analysis
            - Cup-to-disc ratio assessment
            - Retinal nerve fiber analysis
            - Early glaucoma detection
            """)

if __name__ == "__main__":
    main()