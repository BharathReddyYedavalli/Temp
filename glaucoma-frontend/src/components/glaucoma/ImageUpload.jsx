import React from "react";
import { Upload, Image as ImageIcon } from "lucide-react";

export default function ImageUpload({ onFileSelect, imagePreview, hasFile }) {
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) onFileSelect(file);
  };

  const openFileDialog = () => {
    document.getElementById("image-upload").click();
  };

  return (
    <div className="border border-blue-200 bg-white rounded-xl p-6 shadow hover:shadow-md transition">
      <h2 className="text-lg font-semibold text-blue-700 flex items-center gap-2 mb-4">
        <Upload className="w-5 h-5" />
        Upload Fundus Image
      </h2>

      <div
        onClick={openFileDialog}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer ${
          hasFile ? "border-blue-400" : "border-blue-200"
        } hover:border-blue-400 bg-blue-50`}
      >
        {imagePreview ? (
          <img
            src={imagePreview}
            alt="Preview"
            className="mx-auto h-48 object-contain rounded"
          />
        ) : (
          <>
            <div className="flex justify-center mb-4">
              <ImageIcon className="w-12 h-12 text-blue-400" />
            </div>
            <p className="text-blue-600 font-medium">Upload Fundus Image</p>
            <p className="text-sm text-blue-500 mt-1">
              Drag and drop your fundus image here, or click to browse
            </p>
            <p className="text-xs text-blue-400 mt-2">Supports JPEG and PNG formats</p>
          </>
        )}
        <input
          id="image-upload"
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
    </div>
  );
}
