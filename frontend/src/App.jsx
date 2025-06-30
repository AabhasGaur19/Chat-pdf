// import { useState, useEffect } from 'react';
// import FileUpload from './components/FileUpload';
// import ChatInterface from './components/ChatInterface';
// import './styles/animations.css';

// function App() {
//   const [chatId, setChatId] = useState('');
//   const [error, setError] = useState('');

//   useEffect(() => {
//     const storedChatId = sessionStorage.getItem('chat_id');
//     if (storedChatId) {
//       setChatId(storedChatId);
//     }
//   }, []);

//   const handleChatIdUpdate = (newChatId) => {
//     setChatId(newChatId);
//     sessionStorage.setItem('chat_id', newChatId);
//   };

//   return (
//     <div className="min-h-screen bg-black text-white overflow-hidden relative">
//       {/* Animated background */}
//       <div className="fixed inset-0 z-0">
//         <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-blue-900/20 to-pink-900/20"></div>
//         <div className="absolute top-0 left-0 w-full h-full">
//           <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse"></div>
//           <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
//           <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
//         </div>
//       </div>

//       {/* Main content */}
//       <div className="relative z-10">
//         <div className="container mx-auto px-4 py-6 max-w-6xl">
//           {/* Header */}
//           <div className="text-center mb-8 animate-fadeInUp">
//             <h1 className="text-4xl md:text-5xl font-bold mb-3">
//               <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-red-400 bg-clip-text text-transparent">
//                 PDF Chat
//               </span>
//             </h1>
//             <p className="text-lg text-gray-400 max-w-2xl mx-auto">
//               Upload your PDF and start an intelligent conversation
//             </p>
            
//             {chatId && (
//               <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-green-500/20 border border-green-500/30 rounded-full animate-slideDown">
//                 <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
//                 <span className="text-green-300 text-sm font-medium">
//                   Connected: {chatId.substring(0, 8)}...
//                 </span>
//               </div>
//             )}
//           </div>

//           {/* Error Display */}
//           {error && (
//             <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl animate-slideDown">
//               <p className="text-red-300 text-center">{error}</p>
//             </div>
//           )}

//           {/* Main Layout */}
//           <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
//             {/* Left sidebar - File Upload */}
//             <div className="lg:col-span-1">
//               <FileUpload setChatId={handleChatIdUpdate} setError={setError} />
//             </div>
            
//             {/* Right side - Chat Interface */}
//             <div className="lg:col-span-2">
//               <ChatInterface chatId={chatId} setError={setError} />
//             </div>
//           </div>

//           {/* Footer */}
//           <div className="text-center mt-12 text-gray-500 animate-fadeInUp" style={{ animationDelay: '1s' }}>
//             <p>Powered by AI • Built with ❤️</p>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }

// export default App;


import { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';
import './styles/animations.css';

function App() {
  const [chatId, setChatId] = useState('');
  const [error, setError] = useState('');
  const [isPdfUploaded, setIsPdfUploaded] = useState(false);

  useEffect(() => {
    const storedChatId = sessionStorage.getItem('chat_id');
    if (storedChatId) {
      setChatId(storedChatId);
      setIsPdfUploaded(true);
    }
  }, []);

  const handleChatIdUpdate = (newChatId) => {
    setChatId(newChatId);
    setIsPdfUploaded(true);
    sessionStorage.setItem('chat_id', newChatId);
  };

  const handleNewUpload = () => {
    setChatId('');
    setIsPdfUploaded(false);
    sessionStorage.removeItem('chat_id');
    setError('');
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden relative">
      {/* Enhanced animated background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/30 via-blue-900/30 to-pink-900/30"></div>
        <div className="absolute top-0 left-0 w-full h-full">
          {/* Floating particles */}
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-float"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-float-delayed"></div>
          <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-float-slow"></div>
          
          {/* Additional smaller particles */}
          <div className="absolute top-1/6 right-1/3 w-32 h-32 bg-cyan-400/10 rounded-full blur-2xl animate-float-fast"></div>
          <div className="absolute bottom-1/6 left-1/6 w-24 h-24 bg-emerald-400/15 rounded-full blur-xl animate-float"></div>
        </div>
        
        {/* Grid overlay */}
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
      </div>

      {/* Main content */}
      <div className="relative z-10">
        <div className="container mx-auto px-4 py-6 max-w-7xl">
          {/* Enhanced Header */}
          <div className="text-center mb-12 animate-fadeInUp">
            <div className="relative inline-block">
              <h1 className="text-5xl md:text-7xl font-black mb-4 relative">
                <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-red-400 bg-clip-text text-transparent animate-gradient-x">
                  PDF Chat
                </span>
                <div className="absolute -inset-1 bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg blur opacity-20 animate-pulse"></div>
              </h1>
            </div>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Transform your documents into interactive conversations with AI
            </p>
            
            {/* Enhanced status indicator */}
            {isPdfUploaded && (
              <div className="mt-6 inline-flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-emerald-500/20 to-green-500/20 border border-emerald-500/30 rounded-full animate-slideDown backdrop-blur-sm">
                <div className="relative">
                  <div className="w-3 h-3 bg-emerald-400 rounded-full animate-pulse"></div>
                  <div className="absolute -inset-1 bg-emerald-400/50 rounded-full animate-ping"></div>
                </div>
                <span className="text-emerald-300 font-medium">
                  PDF Ready • Start Chatting!
                </span>
                <button
                  onClick={handleNewUpload}
                  className="ml-2 px-3 py-1 text-xs bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 rounded-lg transition-all duration-300"
                >
                  Upload New
                </button>
              </div>
            )}
          </div>

          {/* Error Display with enhanced styling */}
          {error && (
            <div className="mb-8 mx-auto max-w-2xl animate-slideDown">
              <div className="p-4 bg-gradient-to-r from-red-500/20 to-orange-500/20 border border-red-500/30 rounded-2xl backdrop-blur-sm">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse"></div>
                  <p className="text-red-300 font-medium">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Dynamic Layout */}
          <div className={`transition-all duration-1000 ease-in-out ${isPdfUploaded ? 'grid grid-cols-1 xl:grid-cols-4 gap-8' : 'flex justify-center'}`}>
            {/* File Upload Section */}
            <div className={`transition-all duration-1000 ${isPdfUploaded ? 'xl:col-span-1' : 'w-full max-w-md'}`}>
              <FileUpload 
                setChatId={handleChatIdUpdate} 
                setError={setError} 
                isPdfUploaded={isPdfUploaded}
                onNewUpload={handleNewUpload}
              />
            </div>
            
            {/* Chat Interface */}
            {isPdfUploaded && (
              <div className="xl:col-span-3 animate-slideIn">
                <ChatInterface chatId={chatId} setError={setError} />
              </div>
            )}
          </div>

          {/* Enhanced Footer */}
          <div className="text-center mt-16 animate-fadeInUp" style={{ animationDelay: '1.2s' }}>
            <div className="inline-flex items-center gap-2 px-6 py-3 bg-gray-800/30 backdrop-blur-sm rounded-full border border-gray-700/50">
              <div className="w-2 h-2 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full animate-pulse"></div>
              <p className="text-gray-400 text-sm">Powered by AI • Built with passion</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;