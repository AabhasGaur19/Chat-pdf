import { useState ,useEffect} from 'react';
import FileUpload from './components/FileUpload';
import QuestionForm from './components/QuestionForm';
// Main App Component
function App() {
  const [chatId, setChatId] = useState('');
  const [error, setError] = useState('');

  // Load chat_id from localStorage on mount
  useEffect(() => {
    const storedChatId = localStorage.getItem('chat_id');
    if (storedChatId) {
      setChatId(storedChatId);
    }
  }, []);

  // Update chat_id and store in localStorage
  const handleChatIdUpdate = (newChatId) => {
    setChatId(newChatId);
    localStorage.setItem('chat_id', newChatId);
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden relative">
      {/* Custom CSS animations */}
      <style>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes slideDown {
          from {
            opacity: 0;
            max-height: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            max-height: 200px;
            transform: translateY(0);
          }
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }
        
        .animate-fadeInUp {
          animation: fadeInUp 0.6s ease-out forwards;
        }
        
        .animate-slideDown {
          animation: slideDown 0.4s ease-out forwards;
        }
        
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
      `}</style>

      {/* Animated background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-purple-900/20 via-blue-900/20 to-pink-900/20"></div>
        <div className="absolute top-0 left-0 w-full h-full">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
          <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        </div>
      </div>

      {/* Main content */}
      <div className="relative z-10">
        <div className="container mx-auto px-4 py-8 max-w-4xl">
          {/* Header */}
          <div className="text-center mb-12 animate-fadeInUp">
            <h1 className="text-5xl md:text-6xl font-bold mb-4">
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-red-400 bg-clip-text text-transparent animate-pulse">
                PDF Chat
              </span>
            </h1>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Upload your PDF and start an intelligent conversation with your document
            </p>
            
            {chatId && (
              <div className="mt-6 inline-flex items-center gap-2 px-4 py-2 bg-green-500/20 border border-green-500/30 rounded-full animate-slideDown">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-green-300 text-sm font-medium">
                  Chat Active: {chatId.substring(0, 8)}...
                </span>
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl animate-slideDown">
              <p className="text-red-300 text-center">{error}</p>
            </div>
          )}

          {/* Components */}
          <div className="space-y-8">
            <FileUpload setChatId={handleChatIdUpdate} setError={setError} />
            <QuestionForm chatId={chatId} setError={setError} />
          </div>

          {/* Footer */}
          <div className="text-center mt-16 text-gray-500 animate-fadeInUp" style={{ animationDelay: '1s' }}>
            <p>Powered by AI • Built with ❤️</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;