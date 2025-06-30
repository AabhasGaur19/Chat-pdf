import { User, Bot } from 'lucide-react';

function ChatMessage({ message, isUser }) {
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-fadeInUp`}>
      <div className={`flex items-start gap-3 max-w-[80%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        <div className={`p-2 rounded-full ${isUser ? 'bg-gradient-to-r from-blue-500 to-purple-500' : 'bg-gradient-to-r from-green-500 to-emerald-500'}`}>
          {isUser ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-white" />}
        </div>
        <div className={`p-4 rounded-2xl ${isUser 
          ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-br-md' 
          : 'bg-gray-800/70 text-gray-200 rounded-bl-md border border-gray-700/50'
        }`}>
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message}</p>
        </div>
      </div>
    </div>
  );
}

export default ChatMessage;