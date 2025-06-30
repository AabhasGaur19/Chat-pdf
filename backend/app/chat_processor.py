import tempfile
import uuid
import os
import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS as LangChainFAISS
from dotenv import load_dotenv

load_dotenv()

class ChatProcessor:
    def __init__(self):
        # Load environment variables
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        # Load embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize LLM
        self.gemini = GoogleGenerativeAI(model=gemini_model, google_api_key=gemini_api_key)
        
        # Directory for FAISS indices
        self.index_dir = Path("faiss_indices")
        self.index_dir.mkdir(exist_ok=True)
        
        # Dictionary to store in-memory FAISS indices
        self.indices = {}

    def process_pdf(self, file_bytes: bytes) -> Dict[str, str]:
        # Save PDF to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_path = temp_pdf.name

        try:
            # Load and split PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(documents)

            # Create unique chat ID
            chat_id = str(uuid.uuid4())

            # Prepare texts and metadata
            texts = [doc.page_content for doc in chunks]
            metadatas = [{"source": f"page_{i}", "text": texts[i]} for i in range(len(chunks))]
            
            # Choose FAISS index based on number of chunks
            dimension = 384  # Dimension of all-MiniLM-L6-v2 embeddings
            nlist = 100  # Number of clusters for IVFPQ

            if len(chunks) < nlist:
                # Use IndexFlatL2 for small documents
                vectorstore = LangChainFAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
            else:
                # Generate embeddings for large documents
                embeddings = self.embeddings.embed_documents(texts)
                embeddings = np.array(embeddings, dtype=np.float32)

                # Initialize and train IndexIVFPQ
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)  # 8-bit PQ
                index.nprobe = 10  # Number of clusters to search
                index.train(embeddings)
                index.add(embeddings)

                # Create LangChain FAISS vector store from embeddings
                vectorstore = LangChainFAISS.from_embeddings(
                    text_embeddings=list(zip(texts, embeddings)),
                    embedding=self.embeddings,
                    metadatas=metadatas
                )

            # Store in memory and persist to disk
            self.indices[chat_id] = vectorstore
            vectorstore.save_local(str(self.index_dir / chat_id))

            return {"chat_id": chat_id}

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    def ask_question(self, chat_id: str, query: str) -> Dict[str, str]:
        # Load vector store if not in memory
        if chat_id not in self.indices:
            index_path = self.index_dir / chat_id
            if not index_path.exists():
                raise ValueError(f"No index found for chat_id: {chat_id}")
            self.indices[chat_id] = LangChainFAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        # Retrieve vector store
        vectorstore = self.indices[chat_id]

        # Set up retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Set up QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.gemini,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

        # Run query
        result = qa_chain.invoke({"query": query})

        return {"response": result["result"]}

    def delete_index(self, chat_id: str):
        # Remove from memory
        if chat_id in self.indices:
            del self.indices[chat_id]
        
        # Remove from disk
        index_path = self.index_dir / chat_id
        if index_path.exists():
            for file in index_path.glob("*"):
                file.unlink()
            index_path.rmdir()