import { useState } from 'react';
import { Upload, MessageCircle, FileText, Sparkles } from 'lucide-react';

// FileUpload Component
function FileUpload({ setChatId, setError }) {
  const [status, setStatus] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const handleUpload = async (file) => {
    if (!file) {
      setStatus('Please select a PDF file.');
      setError('Please select a PDF file.');
      return;
    }

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/upload_pdf', {
        method: 'POST',
        body: formData,
      });
      const result = await response.json();

      if (response.ok) {
        setStatus(`PDF uploaded successfully! Chat ID: ${result.chat_id}`);
        setChatId(result.chat_id);
        setError('');
      } else {
        setStatus(`Error: ${result.detail}`);
        setError(result.detail);
      }
    } catch (error) {
      setStatus(`Error: ${error.message}`);
      setError(error.message);
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
      <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 hover:border-gray-600/50 transition-all duration-500">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500 animate-pulse">
            <Upload className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            Upload PDF
          </h2>
        </div>

        <div
          className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-300 ${
            isDragging
              ? 'border-purple-400 bg-purple-500/10 scale-105'
              : 'border-gray-600 hover:border-gray-500'
          } ${isUploading ? 'pointer-events-none opacity-50' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {isUploading ? (
            <div className="flex flex-col items-center gap-4">
              <div className="w-12 h-12 border-4 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
              <p className="text-gray-300">Uploading your PDF...</p>
            </div>
          ) : (
            <>
              <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4 animate-bounce" />
              <p className="text-gray-300 mb-4">
                Drag and drop your PDF here, or click to browse
              </p>
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <button
                type="button"
                className="relative px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300 transform hover:scale-105 active:scale-95 shadow-lg shadow-purple-500/25"
              >
                Choose File
              </button>
            </>
          )}
        </div>

        {status && (
          <div className="mt-4 p-4 rounded-lg bg-gray-800/50 border border-gray-700 animate-slideDown">
            <p className="text-gray-300">{status}</p>
          </div>
        )}
      </div>
    </div>
  );
}


export default FileUpload;