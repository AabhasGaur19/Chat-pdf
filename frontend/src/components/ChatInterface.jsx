import { useState, useEffect, useRef } from 'react';
import { MessageCircle, Send, Bot, FileText } from 'lucide-react';
import ChatMessage from './ChatMessage';
import { askQuestion } from '../utils/api';

function ChatInterface({ chatId, setError }) {
  const [question, setQuestion] = useState('');
  const [conversations, setConversations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    const storedConversations = JSON.parse(sessionStorage.getItem(`conversations_${chatId}`) || '[]');
    setConversations(storedConversations);
  }, [chatId]);

  useEffect(() => {
    if (chatId && conversations.length > 0) {
      sessionStorage.setItem(`conversations_${chatId}`, JSON.stringify(conversations));
    }
  }, [conversations, chatId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversations]);

  const handleSubmit = async () => {
    if (!chatId) {
      setError('Please upload a PDF first to get a Chat ID.');
      return;
    }
    
    if (!question.trim()) {
      setError('Please enter a question.');
      return;
    }

    const userMessage = question.trim();
    setQuestion('');
    setIsLoading(true);
    
    setConversations(prev => [...prev, { type: 'user', message: userMessage, timestamp: Date.now() }]);

    try {
      const result = await askQuestion(chatId, userMessage);
      setConversations(prev => [...prev, { type: 'bot', message: result.response, timestamp: Date.now() }]);
      setError('');
    } catch (error) {
      setError(error.message);
      setConversations(prev => prev.slice(0, -1));
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setConversations([]);
    if (chatId) {
      sessionStorage.removeItem(`conversations_${chatId}`);
    }
  };

  return (
    <div className="relative animate-fadeInUp bg-gray-900/50 backdrop-blur-sm rounded-2xl border border-gray-700/50 overflow-hidden" style={{ animationDelay: '0.2s' }}>
      {/* Chat Header */}
      <div className="p-4 border-b border-gray-700/50 bg-gray-800/30">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500">
              <MessageCircle className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Chat with PDF
            </h2>
          </div>
          {conversations.length > 0 && (
            <button
              onClick={clearChat}
              className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
            >
              Clear Chat
            </button>
          )}
        </div>
      </div>

      {/* Chat Messages */}
      <div 
        ref={chatContainerRef}
        className="h-96 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800"
      >
        {conversations.length === 0 && chatId && (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <MessageCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">Start a conversation about your PDF!</p>
            </div>
          </div>
        )}
        
        {conversations.length === 0 && !chatId && (
          <div className="flex items-center justify-center h-full text-gray-400">
            <div className="text-center">
              <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">Upload a PDF to start chatting</p>
            </div>
          </div>
        )}

        {conversations.map((conv, index) => (
          <ChatMessage
            key={`${conv.timestamp}-${index}`}
            message={conv.message}
            isUser={conv.type === 'user'}
          />
        ))}

        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="flex items-start gap-3 max-w-[80%]">
              <div className="p-2 rounded-full bg-gradient-to-r from-green-500 to-emerald-500">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="p-4 rounded-2xl bg-gray-800/70 border border-gray-700/50 rounded-bl-md">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <div className="p-4 border-t border-gray-700/50 bg-gray-800/30">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={chatId ? "Ask a question about your PDF..." : "Upload a PDF first..."}
              disabled={!chatId || isLoading}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            />
          </div>
          <button
            onClick={handleSubmit}
            disabled={isLoading || !chatId || !question.trim()}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 hover:from-blue-700 hover:to-purple-700 active:scale-95"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;