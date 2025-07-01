const API_BASE_URL = 'https://chat-pdf-aqx6.onrender.com';
// const API_BASE_URL = 'http://localhost:8000';

export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload_pdf`, {
    method: 'POST',
    body: formData,
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.detail || 'Failed to upload PDF');
  }

  return result;
};

export const askQuestion = async (chatId, query) => {
  const response = await fetch(`${API_BASE_URL}/ask_question`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chat_id: chatId, query }),
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.detail || 'Failed to get response');
  }

  return result;
};