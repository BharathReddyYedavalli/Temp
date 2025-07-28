# 🔬 GlaucoAI - Advanced Glaucoma Detection System

An AI-powered web application for early glaucoma detection using fundus images. Built with React frontend and FastAPI backend, featuring three state-of-the-art deep learning models.

![GlaucoAI Interface](https://via.placeholder.com/800x400?text=GlaucoAI+Interface)

## 🌟 Features

- **Three AI Models**: EfficientNet, MobileNetV3-Large, and ResNet50
- **Real-time Analysis**: Upload and analyze fundus images instantly
- **Detailed Results**: Probability scores, confidence levels, and risk assessment
- **Responsive Design**: Modern, mobile-friendly interface
- **GPU Acceleration**: CUDA support for faster inference
- **Medical Disclaimer**: Proper medical guidance included

## 🚀 Quick Start

### Option 1: Use Startup Scripts (Easiest)

1. **Start Backend**: Double-click `start_backend.bat`
2. **Start Frontend**: Double-click `start_frontend.bat`
3. **Open Browser**: Go to `http://localhost:5194`

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

#### Frontend Setup
```bash
cd glaucoma-frontend
npm install
npm run dev
```

## 📁 Project Structure

```
GlaucoAI/
├── backend/                 # FastAPI backend server
│   ├── app.py              # Main API server
│   ├── requirements.txt    # Python dependencies
│   └── README.md          # Backend documentation
├── glaucoma-frontend/      # React frontend application
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/         # Main pages
│   │   └── ...
│   ├── package.json       # Node.js dependencies
│   └── ...
├── Notebooks/             # Trained models and research
│   ├── EfficientNet/      # EfficientNet model files
│   ├── MobileNetV3-Large/ # MobileNetV3 model files
│   ├── ResNet50/          # ResNet50 model files
│   └── main_test.ipynb   # Testing notebook
├── start_backend.bat      # Backend startup script
├── start_frontend.bat     # Frontend startup script
└── README.md             # This file
```

## 🤖 AI Models

### EfficientNet-B0
- **Architecture**: Efficient compound scaling
- **Features**: Balanced performance and efficiency
- **Best For**: General-purpose analysis
- **Note**: Automatically handles label mapping (class 0=glaucoma, class 1=normal)

### MobileNetV3-Large
- **Architecture**: Mobile-optimized design
- **Features**: Fast inference, low memory usage
- **Best For**: Real-time applications
- **Note**: Automatically handles label mapping (class 0=glaucoma, class 1=normal)

### ResNet50
- **Architecture**: Deep residual network
- **Features**: High accuracy, proven reliability
- **Best For**: Detailed analysis requiring high precision
- **Note**: Automatically handles label mapping (class 0=glaucoma, class 1=normal)

## 🔧 Technical Stack

### Backend
- **Framework**: FastAPI
- **ML Libraries**: PyTorch, torchvision
- **Image Processing**: Pillow, NumPy
- **Server**: Uvicorn

### Frontend
- **Framework**: React 19
- **Build Tool**: Vite
- **Styling**: Tailwind CSS v4
- **Animations**: Framer Motion
- **Icons**: Lucide React

## 📊 API Endpoints

### GET `/`
Health check and server information

### GET `/models`
List available AI models

### POST `/predict`
Analyze fundus image for glaucoma detection

**Parameters:**
- `image`: Image file (JPG, PNG)
- `model`: Model name (EfficientNet, MobileNetV3, ResNet50)

**Response:**
```json
{
  "prediction": "Glaucoma Detected",
  "confidence": 0.92,
  "probabilities": {
    "normal": 0.08,
    "glaucoma": 0.92
  },
  "model_used": "ResNet50",
  "risk_level": "High"
}
```

## ⚠️ Medical Disclaimer

This application is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Please consult with a qualified ophthalmologist for proper medical evaluation and treatment.

## 🔍 Usage Instructions

1. **Upload Image**: Select a fundus image (retinal photograph)
2. **Choose Model**: Select from EfficientNet, MobileNetV3, or ResNet50
3. **Analyze**: Click "Detect Glaucoma" to run the analysis
4. **Review Results**: Check the prediction, confidence score, and detailed probabilities
5. **Consult Professional**: Follow up with medical professional if needed

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ free space
- **GPU**: NVIDIA GPU with CUDA (optional, for acceleration)

### Software Requirements
- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **npm**: Latest version

## 🛠️ Development

### Adding New Models
1. Train your model using PyTorch
2. Save the model weights as `.pth` file
3. Add model configuration to `backend/app.py`
4. Update frontend model selector

### Customizing UI
- Edit components in `glaucoma-frontend/src/components/`
- Modify styles in Tailwind CSS classes
- Update animations using Framer Motion

## 📈 Performance

- **Model Loading**: ~10-30 seconds (depends on GPU)
- **Inference Time**: 1-3 seconds per image
- **Supported Formats**: JPG, PNG, WebP
- **Max Image Size**: 10MB

## 🚀 Deployment

### Local Deployment
Use the provided startup scripts for easy local deployment.

### Production Deployment
- **Backend**: Deploy to AWS/Azure/GCP with GPU instances
- **Frontend**: Deploy to Vercel/Netlify/GitHub Pages
- **Database**: Add PostgreSQL/MongoDB for user data (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with medical AI regulations in your jurisdiction.

## 🆘 Troubleshooting

### Common Issues

**Backend won't start:**
- Check if Python 3.8+ is installed
- Verify all dependencies are installed: `pip install -r backend/requirements.txt`
- Ensure model files exist in Notebooks folders

**Frontend won't start:**
- Check if Node.js 16+ is installed
- Clear npm cache: `npm cache clean --force`
- Reinstall dependencies: `npm install`

**CORS errors:**
- Ensure backend is running on port 8000
- Check if frontend is accessing `http://localhost:8000`

### Getting Help

1. Check the logs in terminal for error messages
2. Verify file paths and permissions
3. Ensure all dependencies are correctly installed
4. **Model Accuracy**: If predictions seem incorrect, restart the backend server after the label fix
5. Contact support for persistent issues

---

# Glaucoma Detection AI - Streamlit App

A web application for glaucoma detection using deep learning with Grad-CAM visualization and AI-powered medical explanations.

## Features

- 🔍 **AI-Powered Glaucoma Detection**: Upload retinal images for instant analysis
- 🔥 **Grad-CAM Visualization**: See exactly which areas the AI is focusing on
- 🩺 **Medical Explanations**: ChatGPT-powered professional explanations of results
- 📱 **User-Friendly Interface**: Clean, responsive web interface
- ☁️ **Cloud Deployment**: Ready for Streamlit Cloud deployment

## Deployment on Streamlit Cloud

### Quick Deploy
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Set the main file path to `app.py`
6. Add your OpenAI API key in the secrets (see below)
7. Deploy!

### Secrets Configuration
Add the following to your Streamlit Cloud secrets:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

## Local Development

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone <your-repo-url>
cd AI-Glaucoma-Detection-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.streamlit/secrets.toml` file:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. **Upload Image**: Click "Choose a retinal image..." and select a fundus photograph
2. **Analyze**: Click "Analyze Image" to run the AI detection
3. **Review Results**: 
   - View the original image and Grad-CAM overlay
   - Check the prediction confidence score
   - Read the AI-generated medical explanation
4. **Adjust Settings**: Use the sidebar to modify heatmap transparency

## Model Information

The application uses a pre-trained convolutional neural network specifically designed for glaucoma detection from fundus images. The model provides:

- Binary classification (Glaucoma/Normal)
- Confidence scores
- Grad-CAM explanations for model interpretability

## File Structure

```
AI-Glaucoma-Detection-main/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Model files directory
│   └── glaucoma_detection_model.h5
└── .streamlit/           # Streamlit configuration
    └── secrets.toml      # API keys (local only)
```

## Medical Disclaimer

⚠️ **Important**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical concerns.

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI/ML**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Visualization**: Matplotlib
- **AI Integration**: OpenAI GPT-3.5

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source and available under the MIT License.

---

**Made with ❤️ for advancing medical AI research and education**
