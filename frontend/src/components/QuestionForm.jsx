import { useState } from 'react';
import { MessageCircle, Sparkles } from 'lucide-react';

// QuestionForm Component
function QuestionForm({ chatId, setError }) {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (event) => {
    if (event) {
      event.preventDefault();
    }
    if (!chatId) {
      setError('Please upload a PDF first to get a Chat ID.');
      setResponse('');
      return;
    }
    if (!question) {
      setError('Please enter a question.');
      setResponse('');
      return;
    }

    setIsLoading(true);

    try {
      const res = await fetch('http://localhost:8000/ask_question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chat_id: chatId, query: question }),
      });
      const result = await res.json();

      if (res.ok) {
        setResponse(result.response);
        setError('');
      } else {
        setResponse('');
        setError(result.detail);
      }
    } catch (error) {
      setResponse('');
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
    setQuestion('');
  };

  return (
    <div className="relative animate-fadeInUp" style={{ animationDelay: '0.2s' }}>
      <div className="bg-gray-900/50 backdrop-blur-sm rounded-2xl p-8 border border-gray-700/50 hover:border-gray-600/50 transition-all duration-500">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 animate-pulse">
            <MessageCircle className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Ask a Question
          </h2>
        </div>

        <div className="space-y-6">
          <div className="relative group">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your question about the PDF..."
              className="w-full px-6 py-4 bg-gray-800/50 border-2 border-gray-600 rounded-xl text-white placeholder-gray-400 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300"
              style={{
                background: 'linear-gradient(135deg, rgba(17, 24, 39, 0.5) 0%, rgba(31, 41, 55, 0.5) 100%)',
              }}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleSubmit(e);
                }
              }}
            />
            <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-blue-500 via-purple-500 via-pink-500 to-red-500 opacity-0 group-hover:opacity-20 group-focus-within:opacity-20 transition-opacity duration-300 pointer-events-none"></div>
          </div>

          <button
            onClick={handleSubmit}
            disabled={isLoading || !chatId}
            className="relative w-full px-8 py-4 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white rounded-xl font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98] shadow-xl shadow-blue-500/25 overflow-hidden group"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            <div className="relative flex items-center justify-center gap-3">
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  <span>Ask Question</span>
                </>
              )}
            </div>
          </button>
        </div>

        {response && (
          <div className="mt-6 p-6 rounded-xl bg-gradient-to-br from-gray-800/70 to-gray-900/70 border border-gray-600/50 backdrop-blur-sm animate-slideDown">
            <div className="flex items-start gap-3 mb-3">
              <div className="p-1.5 rounded-lg bg-gradient-to-r from-green-500 to-emerald-500 mt-1 animate-pulse">
                <MessageCircle className="w-4 h-4 text-white" />
              </div>
              <h3 className="text-lg font-semibold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                Answer
              </h3>
            </div>
            <div className="pl-10">
              <p className="text-gray-200 leading-relaxed whitespace-pre-wrap">{response}</p>
            </div>
          </div>
        )}

        {!chatId && (
          <div className="mt-4 p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/30 animate-slideDown">
            <p className="text-yellow-300 text-sm">
              💡 Upload a PDF first to start asking questions
            </p>
          </div>
        )}
      </div>
    </div>
  );
}



export default QuestionForm;