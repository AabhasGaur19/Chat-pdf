import { useState, useEffect, useRef } from 'react';
import { MessageCircle, Send, Bot, FileText, Sparkles } from 'lucide-react';
import ChatMessage from './ChatMessage';
import { askQuestion } from '../utils/api';

function ChatInterface({ chatId, setError }) {
  const [question, setQuestion] = useState('');
  const [conversations, setConversations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const inputRef = useRef(null);

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

  useEffect(() => {
    if (chatId && inputRef.current) {
      inputRef.current.focus();
    }
  }, [chatId]);

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
    setIsTyping(true);
    
    setConversations(prev => [...prev, { type: 'user', message: userMessage, timestamp: Date.now() }]);

    try {
      const result = await askQuestion(chatId, userMessage);
      setIsTyping(false);
      setTimeout(() => {
        setConversations(prev => [...prev, { type: 'bot', message: result.response, timestamp: Date.now() }]);
      }, 500);
      setError('');
    } catch (error) {
      setError(error.message);
      setConversations(prev => prev.slice(0, -1));
      setIsTyping(false);
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
    <div className="relative animate-fadeInUp bg-gray-900/50 backdrop-blur-sm rounded-2xl border border-gray-700/50 overflow-hidden hover:border-gray-600/50 transition-all duration-500 hover:shadow-2xl hover:shadow-blue-500/20" style={{ animationDelay: '0.2s' }}>
      {/* Enhanced Chat Header */}
      <div className="p-4 border-b border-gray-700/50 bg-gray-800/30 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 animate-glow-blue">
              <MessageCircle className="w-5 h-5 text-white" />
            </div>
            <h2 className="text-lg font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Chat with PDF
            </h2>
            {chatId && (
              <div className="flex items-center gap-1">
                <Sparkles className="w-4 h-4 text-green-400 animate-twinkle" />
                <span className="text-xs text-green-400 animate-pulse">Ready</span>
              </div>
            )}
          </div>
          {conversations.length > 0 && (
            <button
              onClick={clearChat}
              className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-all duration-300 transform hover:scale-105 active:scale-95"
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
          <div className="flex items-center justify-center h-full text-gray-400 animate-bounceIn">
            <div className="text-center">
              <MessageCircle className="w-12 h-12 mx-auto mb-3 opacity-50 animate-float" />
              <p className="text-sm animate-pulse">Start a conversation about your PDF!</p>
              <div className="flex justify-center gap-1 mt-2">
                <div className="w-2 h-2 bg-blue-400/50 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-purple-400/50 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-pink-400/50 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}
        
        {conversations.length === 0 && !chatId && (
          <div className="flex items-center justify-center h-full text-gray-400 animate-fadeIn">
            <div className="text-center">
              <FileText className="w-12 h-12 mx-auto mb-3 opacity-50 animate-pulse" />
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

        {(isLoading || isTyping) && (
          <div className="flex justify-start mb-4 animate-slideInLeft">
            <div className="flex items-start gap-3 max-w-[80%]">
              <div className="p-2 rounded-full bg-gradient-to-r from-green-500 to-emerald-500 animate-glow-green">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <div className="p-4 rounded-2xl bg-gray-800/70 border border-gray-700/50 rounded-bl-md backdrop-blur-sm">
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

      {/* Enhanced Chat Input */}
      <div className="p-4 border-t border-gray-700/50 bg-gray-800/30 backdrop-blur-sm">
        <div className="flex gap-3">
          <div className="flex-1 relative group">
            <input
              ref={inputRef}
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
              className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed group-hover:bg-gray-800/70 focus:bg-gray-800/80"
            />
            {question && (
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
              </div>
            )}
          </div>
          <button
            onClick={handleSubmit}
            disabled={isLoading || !chatId || !question.trim()}
            className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 hover:from-blue-700 hover:to-purple-700 active:scale-95 transform hover:scale-105 animate-glow-blue"
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

