// import { useState } from 'react';
// import { Upload, FileText } from 'lucide-react';
// import { uploadPDF } from '../utils/api';

// function FileUpload({ setChatId, setError }) {
//   const [status, setStatus] = useState('');
//   const [isDragging, setIsDragging] = useState(false);
//   const [isUploading, setIsUploading] = useState(false);

//   const handleUpload = async (file) => {
//     if (!file) {
//       setStatus('Please select a PDF file.');
//       setError('Please select a PDF file.');
//       return;
//     }

//     setIsUploading(true);
//     try {
//       const result = await uploadPDF(file);
//       setStatus(`PDF uploaded successfully! Chat ID: ${result.chat_id}`);
//       setChatId(result.chat_id);
//       setError('');
//     } catch (error) {
//       setStatus(`Error: ${error.message}`);
//       setError(error.message);
//     } finally {
//       setIsUploading(false);
//     }
//   };

//   const handleFileChange = (event) => {
//     const file = event.target.files[0];
//     handleUpload(file);
//   };

//   const handleDrop = (event) => {
//     event.preventDefault();
//     setIsDragging(false);
//     const file = event.dataTransfer.files[0];
//     handleUpload(file);
//   };

//   const handleDragOver = (event) => {
//     event.preventDefault();
//     setIsDragging(true);
//   };

//   const handleDragLeave = () => {
//     setIsDragging(false);
//   };

//   return (
//     <div className="relative animate-fadeInUp">
//       <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl p-6 border border-gray-700/50">
//         <div className="flex items-center gap-3 mb-4">
//           <div className="p-2 rounded-lg bg-gradient-to-r from-purple-500 to-pink-500">
//             <Upload className="w-5 h-5 text-white" />
//           </div>
//           <h2 className="text-lg font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
//             Upload PDF
//           </h2>
//         </div>

//         <div
//           className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all duration-300 ${
//             isDragging
//               ? 'border-purple-400 bg-purple-500/10'
//               : 'border-gray-600 hover:border-gray-500'
//           } ${isUploading ? 'pointer-events-none opacity-50' : ''}`}
//           onDrop={handleDrop}
//           onDragOver={handleDragOver}
//           onDragLeave={handleDragLeave}
//         >
//           {isUploading ? (
//             <div className="flex flex-col items-center gap-3">
//               <div className="w-8 h-8 border-4 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
//               <p className="text-gray-300 text-sm">Uploading...</p>
//             </div>
//           ) : (
//             <>
//               <FileText className="w-12 h-12 text-gray-400 mx-auto mb-3" />
//               <p className="text-gray-300 mb-3 text-sm">
//                 Drag and drop your PDF here
//               </p>
//               <input
//                 type="file"
//                 accept=".pdf"
//                 onChange={handleFileChange}
//                 className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
//               />
//               <button
//                 type="button"
//                 className="px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg text-sm font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300"
//               >
//                 Choose File
//               </button>
//             </>
//           )}
//         </div>

//         {status && (
//           <div className="mt-3 p-3 rounded-lg bg-gray-800/50 border border-gray-700">
//             <p className="text-gray-300 text-sm">{status}</p>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// }

// export default FileUpload;


import { useState } from 'react';
import { Upload, FileText, CheckCircle, RotateCcw, Sparkles } from 'lucide-react';
import { uploadPDF } from '../utils/api';

function FileUpload({ setChatId, setError, isPdfUploaded, onNewUpload }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleUpload = async (file) => {
    if (!file) {
      setError('Please select a PDF file.');
      return;
    }

    if (file.type !== 'application/pdf') {
      setError('Please select a valid PDF file.');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    
    // Simulate upload progress
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + Math.random() * 15;
      });
    }, 200);

    try {
      const result = await uploadPDF(file);
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      setTimeout(() => {
        setChatId(result.chat_id);
        setError('');
      }, 500);
    } catch (error) {
      clearInterval(progressInterval);
      setUploadProgress(0);
      setError(error.message);
    } finally {
      setTimeout(() => {
        setIsUploading(false);
        setUploadProgress(0);
      }, 1000);
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

  if (isPdfUploaded) {
    return (
      <div className="relative animate-fadeInUp">
        <div className="bg-gradient-to-br from-gray-900/60 to-gray-800/60 backdrop-blur-md rounded-3xl p-6 border border-emerald-500/30 shadow-2xl">
          <div className="text-center">
            <div className="relative inline-block mb-4">
              <div className="p-4 rounded-2xl bg-gradient-to-r from-emerald-500 to-green-500 shadow-lg">
                <CheckCircle className="w-8 h-8 text-white" />
              </div>
              <div className="absolute -inset-2 bg-gradient-to-r from-emerald-500 to-green-500 rounded-2xl blur opacity-30 animate-pulse"></div>
            </div>
            
            <h3 className="text-xl font-bold text-emerald-400 mb-2">PDF Loaded!</h3>
            <p className="text-gray-300 text-sm mb-6">Your document is ready for conversation</p>
            
            <button
              onClick={onNewUpload}
              className="group inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300 shadow-lg hover:shadow-xl active:scale-95"
            >
              <RotateCcw className="w-4 h-4 group-hover:rotate-180 transition-transform duration-300" />
              Upload New PDF
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative animate-fadeInUp">
      <div className="bg-gradient-to-br from-gray-900/60 to-gray-800/60 backdrop-blur-md rounded-3xl p-8 border border-gray-700/50 shadow-2xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="relative">
            <div className="p-3 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 shadow-lg">
              <Upload className="w-6 h-6 text-white" />
            </div>
            <Sparkles className="absolute -top-1 -right-1 w-4 h-4 text-yellow-400 animate-pulse" />
          </div>
          <div>
            <h2 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              Upload Your PDF
            </h2>
            <p className="text-gray-400 text-sm">Drag & drop or click to select</p>
          </div>
        </div>

        <div
          className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-500 ${
            isDragging
              ? 'border-purple-400 bg-purple-500/20 scale-105'
              : 'border-gray-600 hover:border-purple-500/50 hover:bg-purple-500/5'
          } ${isUploading ? 'pointer-events-none' : 'cursor-pointer'}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          {isUploading ? (
            <div className="flex flex-col items-center gap-4">
              <div className="relative">
                <div className="w-16 h-16 border-4 border-purple-400/30 border-t-purple-400 rounded-full animate-spin"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <FileText className="w-6 h-6 text-purple-400" />
                </div>
              </div>
              <div className="w-full max-w-xs">
                <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-purple-500 to-pink-500 h-full transition-all duration-300 ease-out"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="text-gray-300 text-sm mt-2">Uploading... {Math.round(uploadProgress)}%</p>
              </div>
            </div>
          ) : (
            <>
              <div className="relative mb-4">
                <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-20 h-20 border-2 border-gray-600 border-dashed rounded-xl opacity-30"></div>
                </div>
              </div>
              
              <p className="text-gray-300 mb-4 text-lg font-medium">
                Drop your PDF here
              </p>
              <p className="text-gray-500 text-sm mb-6">
                or click to browse files
              </p>
              
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              
              <div className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-medium hover:from-purple-700 hover:to-pink-700 transition-all duration-300 shadow-lg hover:shadow-xl active:scale-95">
                <Upload className="w-5 h-5" />
                Choose File
              </div>
            </>
          )}
        </div>

        <div className="mt-6 flex items-center justify-center gap-2 text-gray-500 text-xs">
          <div className="w-1 h-1 bg-gray-500 rounded-full"></div>
          <span>Supports PDF files up to 10MB</span>
          <div className="w-1 h-1 bg-gray-500 rounded-full"></div>
        </div>
      </div>
    </div>
  );
}

export default FileUpload;