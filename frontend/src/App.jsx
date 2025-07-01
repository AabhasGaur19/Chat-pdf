import { useState, useEffect } from 'react';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';
import './styles/animations.css';

function App() {
  const [chatId, setChatId] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    const storedChatId = sessionStorage.getItem('chat_id');
    if (storedChatId) {
      setChatId(storedChatId);
    }
  }, []);

  const handleChatIdUpdate = (newChatId) => {
    setChatId(newChatId);
    sessionStorage.setItem('chat_id', newChatId);
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden relative">
      {/* Enhanced animated background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-blue-900/20 to-pink-900/20"></div>
        <div className="absolute top-0 left-0 w-full h-full">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-floating"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-floating-delayed"></div>
          <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl animate-floating-slow"></div>
          <div className="absolute top-1/3 right-1/3 w-64 h-64 bg-cyan-500/8 rounded-full blur-2xl animate-floating-reverse"></div>
          <div className="absolute bottom-1/3 left-1/3 w-80 h-80 bg-indigo-500/8 rounded-full blur-3xl animate-floating-fast"></div>
        </div>
        
        {/* Particle effect */}
        <div className="absolute inset-0">
          {[...Array(20)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-white/20 rounded-full animate-twinkle"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 3}s`,
                animationDuration: `${2 + Math.random() * 2}s`
              }}
            />
          ))}
        </div>
      </div>

      {/* Main content */}
      <div className="relative z-10">
        <div className="container mx-auto px-4 py-6 max-w-6xl">
          {/* Enhanced Header */}
          <div className="text-center mb-8 animate-fadeInUp">
            <h1 className="text-4xl md:text-5xl font-bold mb-3 animate-gradient-text">
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-red-400 bg-clip-text text-transparent animate-shimmer">
                PDF Chat
              </span>
            </h1>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto animate-typewriter">
              Upload your PDF and start an intelligent conversation
            </p>
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl animate-slideDown backdrop-blur-sm">
              <p className="text-red-300 text-center">{error}</p>
            </div>
          )}

          {/* Main Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left sidebar - File Upload */}
            <div className="lg:col-span-1 animate-slideInLeft">
              <FileUpload setChatId={handleChatIdUpdate} setError={setError} />
            </div>
            
            {/* Right side - Chat Interface */}
            <div className="lg:col-span-2 animate-slideInRight">
              <ChatInterface chatId={chatId} setError={setError} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;