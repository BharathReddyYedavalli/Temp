import React, { useState } from 'react';
import { Loader2, Eye, Upload, Brain, AlertTriangle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

import ImageUpload from "../components/glaucoma/ImageUpload";
import ModelSelector from "../components/glaucoma/ModelSelector";
import ResultsDisplay from "../components/glaucoma/ResultsDisplay";

export default function GlaucomaDetector() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('ResNet50');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setResults(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleDetectGlaucoma = async () => {
    if (!selectedFile) {
      setError("Please upload a fundus image first");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('model', selectedModel);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `API Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze the image. Please ensure the backend server is running on http://localhost:8000');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <motion.header 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-blue-600 rounded-2xl shadow-lg">
              <Eye className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Glaucoma Detection
            </h1>
          </div>
          <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
            Advanced AI-powered analysis of fundus images for early glaucoma detection
          </p>
        </motion.header>

        {/* Error Alert */}
        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="mb-8 bg-red-50 border border-red-200 p-4 rounded-lg flex items-start gap-3 text-red-800"
          >
            <AlertTriangle className="h-5 w-5 mt-1" />
            <div>
              <strong className="block font-semibold mb-1">Analysis Error</strong>
              <p>{error}</p>
            </div>
          </motion.div>
        )}

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
            >
              <ImageUpload 
                onFileSelect={handleFileSelect}
                imagePreview={imagePreview}
                hasFile={!!selectedFile}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <ModelSelector 
                selectedModel={selectedModel}
                onModelChange={setSelectedModel}
              />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="flex gap-3"
            >
              <button
                onClick={handleDetectGlaucoma}
                disabled={!selectedFile || isAnalyzing}
                className="flex-1 h-14 text-lg font-semibold bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 shadow-lg hover:shadow-xl text-white rounded transition-all duration-300 flex items-center justify-center"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Brain className="w-5 h-5 mr-3" />
                    Detect Glaucoma
                  </>
                )}
              </button>
              {(selectedFile || results) && (
                <button
                  onClick={resetAnalysis}
                  className="h-14 px-6 border border-slate-300 hover:border-slate-400 rounded"
                >
                  Reset
                </button>
              )}
            </motion.div>
          </div>

          {/* Right Column - Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <AnimatePresence mode="wait">
              {results ? (
                <ResultsDisplay 
                  results={results}
                  selectedModel={selectedModel}
                  imagePreview={imagePreview}
                />
              ) : (
                <div className="h-full border border-slate-200 rounded-lg shadow-lg p-8 flex flex-col items-center justify-center min-h-[400px] text-center bg-white">
                  <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mb-6">
                    <Brain className="w-10 h-10 text-slate-400" />
                  </div>
                  <h3 className="text-xl font-semibold text-slate-700 mb-3">
                    Ready for Analysis
                  </h3>
                  <p className="text-slate-500 max-w-sm">
                    Upload a fundus image and select a model to begin glaucoma detection analysis
                  </p>
                </div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>

        {/* Info Section */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-16 text-center"
        >
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 shadow-lg border border-slate-200">
            <h2 className="text-2xl font-bold text-slate-800 mb-4">
              Advanced AI Models
            </h2>
            <div className="grid md:grid-cols-3 gap-6 text-left">
              <div className="p-4 bg-blue-50 rounded-xl">
                <h3 className="font-semibold text-blue-800 mb-2">MobileNetV3</h3>
                <p className="text-sm text-blue-600">Optimized for mobile deployment with efficient architecture</p>
              </div>
              <div className="p-4 bg-green-50 rounded-xl">
                <h3 className="font-semibold text-green-800 mb-2">ResNet50</h3>
                <p className="text-sm text-green-600">Deep residual network with proven accuracy</p>
              </div>
              <div className="p-4 bg-purple-50 rounded-xl">
                <h3 className="font-semibold text-purple-800 mb-2">EfficientNet</h3>
                <p className="text-sm text-purple-600">Balanced efficiency and performance</p>
              </div>
            </div>
          </div>
        </motion.section>
      </div>
    </div>
  );
}
