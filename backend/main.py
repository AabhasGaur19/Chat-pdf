# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from app.chat_processor import ChatProcessor  # Updated import
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # Allow CORS for frontend communication
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize ChatProcessor
# chat_processor = ChatProcessor()

# # Pydantic model for question request
# class QuestionRequest(BaseModel):
#     chat_id: str
#     query: str

# @app.post("/upload_pdf")
# async def upload_pdf(file: UploadFile = File(...)):
#     if not file.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
#     try:
#         file_bytes = await file.read()
#         result = chat_processor.process_pdf(file_bytes)
#         return result
#     except Exception as e:
#         logger.error(f"Error processing PDF: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# @app.post("/ask_question")
# async def ask_question(request: QuestionRequest):
#     try:
#         result = chat_processor.ask_question(request.chat_id, request.query)
#         return result
#     except Exception as e:
#         logger.error(f"Error answering question: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.chat_processor import ChatProcessor  # Updated import
import logging
import os
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChatProcessor
chat_processor = ChatProcessor()

# Pydantic model for question request
class QuestionRequest(BaseModel):
    chat_id: str
    query: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        file_bytes = await file.read()
        result = chat_processor.process_pdf(file_bytes)
        return result
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask_question")
async def ask_question(request: QuestionRequest):
    try:
        result = chat_processor.ask_question(request.chat_id, request.query)
        return result
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI server is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)