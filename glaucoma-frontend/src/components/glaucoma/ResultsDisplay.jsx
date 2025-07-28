import React from "react";
import { CheckCircle, XCircle, Info } from "lucide-react";
import { motion } from "framer-motion";

export default function ResultsDisplay({ results, selectedModel, imagePreview }) {
  const isGlaucoma = results?.prediction?.toLowerCase().includes("glaucoma");
  const confidence = parseFloat(results?.confidence || 0);
  const riskLevel = results?.risk_level || "Unknown";
  const probabilities = results?.probabilities || {};

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="border border-slate-200 bg-white rounded-xl shadow-lg p-6 space-y-6"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-slate-800 flex items-center gap-2">
          <Info className="w-5 h-5 text-blue-600" />
          Analysis Results
        </h2>
        <span className="text-sm text-slate-500">Model: {selectedModel}</span>
      </div>

      {imagePreview && (
        <img
          src={imagePreview}
          alt="Analyzed Fundus"
          className="w-full h-60 object-cover rounded-lg border border-slate-100"
        />
      )}

      <div
        className={`flex items-center gap-3 p-4 rounded-lg border ${
          isGlaucoma
            ? "bg-red-50 border-red-200 text-red-700"
            : "bg-green-50 border-green-200 text-green-700"
        }`}
      >
        {isGlaucoma ? (
          <XCircle className="w-6 h-6" />
        ) : (
          <CheckCircle className="w-6 h-6" />
        )}
        <div>
          <p className="font-semibold">{results?.prediction || "No Result"}</p>
          <p className="text-sm opacity-80">
            Confidence: {(confidence * 100).toFixed(1)}%
          </p>
          <p className="text-xs opacity-70">
            Risk Level: {riskLevel}
          </p>
        </div>
      </div>

      {/* Detailed Probabilities */}
      {probabilities && (
        <div className="space-y-3">
          <p className="text-sm text-slate-600 font-medium">Detailed Analysis</p>
          
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-600">Normal</span>
              <span className="text-sm font-medium">{(probabilities.normal * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500"
                style={{ width: `${probabilities.normal * 100}%` }}
              />
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-sm text-slate-600">Glaucoma</span>
              <span className="text-sm font-medium">{(probabilities.glaucoma * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-red-500"
                style={{ width: `${probabilities.glaucoma * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Medical Disclaimer */}
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <p className="text-xs text-amber-800">
          <strong>⚠️ Medical Disclaimer:</strong> This AI analysis is for educational and research purposes only. 
          It should not be used as a substitute for professional medical diagnosis. 
          Please consult with a qualified ophthalmologist for proper medical evaluation.
        </p>
      </div>
    </motion.div>
  );
}
