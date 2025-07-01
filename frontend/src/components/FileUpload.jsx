import { useState } from 'react';
import { Upload, FileText, CheckCircle } from 'lucide-react';
import { uploadPDF } from '../utils/api';

function FileUpload({ setChatId, setError }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isUploaded, setIsUploaded] = useState(false);

  const handleUpload = async (file) => {
    if (!file) {
      setError('Please select a PDF file.');
      return;
    }

    setIsUploading(true);
    setIsUploaded(false);
    try {
      const result = await uploadPDF(file);
      setChatId(result.chat_id);
      setError('');
      setIsUploaded(true);
      
      // Reset uploaded state after 3 seconds
      setTimeout(() => {
        setIsUploaded(false);
      }, 3000);
    } catch (error) {
      setError(error.message);
      setIsUploaded(false);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    handleUpload(file);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    handleUpload(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <div className="relative animate-fadeInUp">
      <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl p-6 border border-gray-700/50 hover:border-gray-600/50 transition-all duration-500 hover:shadow-2xl hover:shadow-purple-500/20">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 animate-glow">
            <Upload className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-lg font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            Upload PDF
          </h2>
        </div>

        <div
          className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all duration-500 transform ${
            isDragging
              ? 'border-purple-400 bg-purple-500/20 scale-105'
              : isUploaded
              ? 'border-green-400 bg-green-500/20'
              : 'border-g ray-600 hover:border-gray-500 hover:bg-gray-800/30'
          } ${isUploading ? 'pointer-events-none opacity-50' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {isUploading ? (
            <div className="flex flex-col items-center gap-3">
              <div className="w-8 h-8 border-4 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-gray-300 text-sm animate-pulse">Processing your PDF...</p>
            </div>
          ) : isUploaded ? (
            <div className="flex flex-col items-center gap-3 animate-bounceIn">
              <CheckCircle className="w-12 h-12 text-green-400 animate-checkmark" />
              <p className="text-green-300 text-sm font-medium">PDF uploaded successfully!</p>
            </div>
          ) : (
            <>
              <FileText className={`w-12 h-12 text-gray-400 mx-auto mb-3 transition-all duration-300 ${isDragging ? 'text-purple-400 scale-110' : 'hover:text-gray-300'}`} />
              <p className="text-gray-300 mb-3 text-sm">
                Drag and drop your PDF here
              </p>
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <button
                type="button"
                className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg text-sm font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300 transform hover:scale-105 active:scale-95 animate-pulse-button"
              >
                Choose File
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default FileUpload;






