import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import openai
import os
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Glaucoma Detection AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1e88e5;
        margin-bottom: 2rem;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .glaucoma-positive {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .glaucoma-negative {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    .confidence-score {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI API
openai.api_key = st.secrets.get("OPENAI_API_KEY", "")

@st.cache_resource
def load_glaucoma_model():
    """Load the pre-trained glaucoma detection model"""
    try:
        model = load_model('/Users/bharathreddy/Downloads/AI-Glaucoma-Detection-main/models/glaucoma_detection_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model prediction"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize image
    img_resized = cv2.resize(img_array, target_size)
    
    # Normalize pixel values
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the output neuron (top predicted or chosen)
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def create_superimposed_img(img, heatmap, alpha=0.4):
    """Create superimposed image with heatmap overlay"""
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Create superimposed image
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    return superimposed_img.astype(np.uint8)

def get_medical_explanation(prediction, confidence):
    """Get medical explanation from ChatGPT API"""
    if not openai.api_key:
        return "Medical explanation not available. Please configure OpenAI API key."
    
    try:
        diagnosis = "positive for glaucoma" if prediction == 1 else "negative for glaucoma"
        
        prompt = f"""
        As a medical AI assistant, provide a professional explanation for a glaucoma detection result.
        
        Result: {diagnosis}
        Confidence: {confidence:.2%}
        
        Please provide:
        1. What this result means for the patient
        2. Recommended next steps
        3. Important disclaimers about AI diagnosis
        
        Keep the response professional, clear, and appropriate for patients to understand.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating medical explanation: {str(e)}"

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ Glaucoma Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a retinal image. The model will predict whether it is glaucoma or normal and show a Grad-CAM heatmap as explanation.</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_glaucoma_model()
    if model is None:
        st.error("Could not load the glaucoma detection model. Please check the model file.")
        return
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    alpha_value = st.sidebar.slider("Heatmap Transparency", 0.1, 1.0, 0.4, 0.1)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a retinal image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a fundus/retinal image for glaucoma detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Uploaded Retinal Image")
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Process image when button is clicked
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                # Preprocess image
                processed_img = preprocess_image(image)
                
                # Make prediction
                prediction = model.predict(processed_img)
                predicted_class = int(prediction[0][0] > 0.5)
                confidence = float(prediction[0][0]) if predicted_class == 1 else float(1 - prediction[0][0])
                
                # Generate Grad-CAM
                try:
                    # Find the last convolutional layer
                    last_conv_layer_name = None
                    for layer in reversed(model.layers):
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            last_conv_layer_name = layer.name
                            break
                    
                    if last_conv_layer_name:
                        heatmap = make_gradcam_heatmap(processed_img, model, last_conv_layer_name)
                        
                        # Create superimposed image
                        original_img = np.array(image.resize((224, 224)))
                        superimposed_img = create_superimposed_img(original_img, heatmap, alpha_value)
                        
                        with col2:
                            st.subheader("🔥 Grad-CAM Heatmap Overlay")
                            st.image(superimposed_img, caption="Grad-CAM Visualization", use_column_width=True)
                    else:
                        st.error("Could not find convolutional layer for Grad-CAM generation.")
                
                except Exception as e:
                    st.error(f"Error generating Grad-CAM: {str(e)}")
                
                # Display prediction results
                st.subheader("📊 Analysis Results")
                
                if predicted_class == 1:
                    st.markdown(f"""
                    <div class="prediction-box glaucoma-positive">
                        <h3>⚠️ GLAUCOMA DETECTED</h3>
                        <div class="confidence-score">Confidence: {confidence:.2%}</div>
                        <p>The AI model detected signs consistent with glaucoma in this retinal image.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box glaucoma-negative">
                        <h3>✅ NORMAL RETINA</h3>
                        <div class="confidence-score">Confidence: {confidence:.2%}</div>
                        <p>The AI model did not detect signs of glaucoma in this retinal image.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Medical explanation
                st.subheader("🩺 Medical Explanation")
                with st.spinner("Generating medical explanation..."):
                    explanation = get_medical_explanation(predicted_class, confidence)
                    st.write(explanation)
                
                # Important disclaimer
                st.warning("""
                **Important Medical Disclaimer:**
                This AI tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical concerns and before making any medical decisions.
                """)
    
    else:
        st.info("👆 Please upload a retinal image to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • Powered by TensorFlow & OpenAI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()