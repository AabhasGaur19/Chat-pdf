import { User, Bot } from 'lucide-react';

function ChatMessage({ message, isUser }) {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-messageSlideIn`}>
      <div className={`flex items-start gap-3 max-w-[80%] ${isUser ? 'flex-row-reverse' : 'flex-row'} group`}>
        <div className={`p-2 rounded-full transition-all duration-300 group-hover:scale-110 ${
          isUser 
            ? 'bg-gradient-to-r from-blue-500 to-purple-500 animate-glow-blue' 
            : 'bg-gradient-to-r from-green-500 to-emerald-500 animate-glow-green'
        }`}>
          {isUser ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-white" />}
        </div>
        <div className={`p-4 rounded-2xl transition-all duration-300 transform hover:scale-[1.02] ${isUser 
          ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-br-md hover:shadow-lg hover:shadow-blue-500/30' 
          : 'bg-gray-800/70 text-gray-200 rounded-bl-md border border-gray-700/50 hover:border-gray-600/50 hover:bg-gray-800/90 backdrop-blur-sm'
        }`}>
          <p className="text-sm leading-relaxed whitespace-pre-wrap animate-typeText">{message}</p>
        </div>
      </div>
    </div>
  );
}

export default ChatMessage;
