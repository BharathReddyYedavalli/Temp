import React from "react";
import { Brain, Cpu, Zap } from "lucide-react";

const modelInfo = {
  MobileNetV3: {
    icon: Zap,
    color: "bg-green-100 text-green-800 border-green-300",
    description: "Lightweight and fast",
    features: ["Mobile optimized", "Fast inference", "Low memory"],
  },
  ResNet50: {
    icon: Brain,
    color: "bg-blue-100 text-blue-800 border-blue-300",
    description: "High accuracy",
    features: ["Deep architecture", "Proven accuracy", "Robust features"],
  },
  EfficientNet: {
    icon: Cpu,
    color: "bg-purple-100 text-purple-800 border-purple-300",
    description: "Balanced performance",
    features: ["Efficient scaling", "Balanced speed/accuracy", "Optimized design"],
  },
};

export default function ModelSelector({ selectedModel, onModelChange }) {
  return (
    <div className="border border-slate-200 bg-white rounded-xl p-6 shadow hover:shadow-md transition">
      <h2 className="text-lg font-semibold text-slate-700 mb-4 flex items-center gap-2">
        <Brain className="w-5 h-5 text-purple-600" />
        Select AI Model
      </h2>

      <select
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        className="w-full border border-slate-300 rounded-lg px-4 py-3 mb-5 text-slate-700 text-base focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="" disabled>
          Choose a model
        </option>
        {Object.keys(modelInfo).map((model) => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>

      {selectedModel && (
        <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              {React.createElement(modelInfo[selectedModel].icon, {
                className: "w-5 h-5 text-slate-600",
              })}
              <h4 className="font-semibold text-slate-800">{selectedModel}</h4>
            </div>
            <span
              className={`px-2 py-1 text-xs font-medium rounded ${modelInfo[selectedModel].color}`}
            >
              {modelInfo[selectedModel].description}
            </span>
          </div>

          <div className="flex flex-wrap gap-2">
            {modelInfo[selectedModel].features.map((feature, index) => (
              <span
                key={index}
                className="px-3 py-1 border border-slate-300 text-xs text-slate-700 rounded-full bg-white"
              >
                {feature}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
