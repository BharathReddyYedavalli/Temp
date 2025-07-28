# 🔬 GlaucoAI - Advanced Glaucoma Detection System

An AI-powered web application for early glaucoma detection using fundus images. Built with **Streamlit** for a simple, intuitive interface featuring simulated AI models with **Grad-CAM visualization**.

## 🌟 Features

- **Demo AI Models**: EfficientNet, MobileNetV3-Large, and ResNet50 simulations
- **Grad-CAM Visualization**: See exactly where the AI focuses attention
- **Real-time Analysis**: Upload and analyze fundus images instantly
- **Intuitive Interface**: Simple drag-and-drop image upload
- **Cross-platform**: Works on any device with a web browser
- **Educational**: Perfect for learning about medical AI

## 🚀 Quick Start

**Live Demo**: [https://glaucoai.streamlit.app](https://glaucoai.streamlit.app)

Or run locally:
```bash
# Clone the repository
git clone https://github.com/BharathReddyYedavalli/AI-Glaucoma-Detection.git
cd AI-Glaucoma-Detection

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

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

**Made with ❤️ for advancing medical AI research and education**
