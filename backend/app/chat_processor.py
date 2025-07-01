# import tempfile
# import uuid
# import os
# import faiss
# import numpy as np
# import gc
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS as LangChainFAISS
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# class ChatProcessor:
#     def __init__(self):
#         # Load environment variables
#         gemini_api_key = os.getenv("GEMINI_API_KEY")
#         gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
#         # Load embedding model with optimized settings
#         model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
#         encode_kwargs = {'normalize_embeddings': True}
        
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs=model_kwargs,
#             encode_kwargs=encode_kwargs
#         )
        
#         # Initialize LLM with optimized settings
#         self.gemini = GoogleGenerativeAI(
#             model=gemini_model, 
#             google_api_key=gemini_api_key,
#             temperature=0.1,
#             max_output_tokens=2048
#         )
        
#         # Directory for FAISS indices
#         self.index_dir = Path("faiss_indices")
#         self.index_dir.mkdir(exist_ok=True)
        
#         # Dictionary to store in-memory FAISS indices (with size limit)
#         self.indices = {}
#         self.max_indices_in_memory = 3  # Limit memory usage
        
#         # Custom prompt template for better responses
#         self.custom_prompt = PromptTemplate(
#             template="""Use the following pieces of context to answer the question at the end. 
#             If you don't know the answer, just say that you don't know, don't try to make up an answer.
#             Try to provide comprehensive answers based on the context provided.

#             Context:
#             {context}

#             Question: {question}
            
#             Answer:""",
#             input_variables=["context", "question"]
#         )

#     def _manage_memory(self):
#         """Remove oldest indices from memory if limit exceeded"""
#         if len(self.indices) > self.max_indices_in_memory:
#             # Remove oldest entry (simple FIFO strategy)
#             oldest_chat_id = next(iter(self.indices))
#             del self.indices[oldest_chat_id]
#             gc.collect()  # Force garbage collection

#     def _get_optimal_chunk_size(self, total_chars: int) -> Tuple[int, int]:
#         """Determine optimal chunk size based on document size"""
#         if total_chars < 10000:  # Small documents
#             return 800, 200
#         elif total_chars < 50000:  # Medium documents
#             return 1000, 250
#         elif total_chars < 200000:  # Large documents
#             return 1200, 300
#         else:  # Very large documents
#             return 1500, 400

#     def _create_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
#         """Create optimized FAISS index based on data size"""
#         n_embeddings, dimension = embeddings.shape
        
#         if n_embeddings < 1000:
#             # Use simple flat index for small datasets
#             index = faiss.IndexFlatL2(dimension)
#         elif n_embeddings < 10000:
#             # Use IVF index for medium datasets
#             nlist = min(int(np.sqrt(n_embeddings)), 100)
#             quantizer = faiss.IndexFlatL2(dimension)
#             index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
#             index.nprobe = min(10, nlist)
#         else:
#             # Use IVF-PQ for large datasets
#             nlist = min(int(np.sqrt(n_embeddings)), 1000)
#             quantizer = faiss.IndexFlatL2(dimension)
#             # Use 8-bit quantization with 8 sub-quantizers
#             index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
#             index.nprobe = min(20, nlist)
        
#         # Train index if needed
#         if hasattr(index, 'train'):
#             logger.info(f"Training FAISS index with {n_embeddings} vectors...")
#             index.train(embeddings)
        
#         # Add vectors to index
#         index.add(embeddings)
#         logger.info(f"Created FAISS index: {type(index).__name__} with {n_embeddings} vectors")
        
#         return index

#     def process_pdf(self, file_bytes: bytes, chunk_overlap_ratio: float = 0.2) -> Dict[str, str]:
#         """Process PDF with optimized chunking and indexing"""
#         # Save PDF to temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#             temp_pdf.write(file_bytes)
#             temp_path = temp_pdf.name

#         try:
#             logger.info("Loading PDF document...")
#             # Load PDF
#             loader = PyPDFLoader(temp_path)
#             documents = loader.load()
            
#             if not documents:
#                 raise ValueError("No content found in PDF")
            
#             # Calculate total characters for optimization
#             total_chars = sum(len(doc.page_content) for doc in documents)
#             logger.info(f"PDF loaded: {len(documents)} pages, {total_chars} total characters")
            
#             # Get optimal chunk size
#             chunk_size, chunk_overlap = self._get_optimal_chunk_size(total_chars)
#             logger.info(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            
#             # Split documents with optimized parameters
#             splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=chunk_size,
#                 chunk_overlap=chunk_overlap,
#                 length_function=len,
#                 separators=["\n\n", "\n", " ", ""]
#             )
#             chunks = splitter.split_documents(documents)
#             logger.info(f"Created {len(chunks)} chunks")

#             # Create unique chat ID
#             chat_id = str(uuid.uuid4())

#             # Prepare texts and enhanced metadata
#             texts = [doc.page_content for doc in chunks]
#             metadatas = []
#             for i, doc in enumerate(chunks):
#                 # Extract page number from document metadata
#                 page_num = doc.metadata.get('page', 0)
#                 source_info = doc.metadata.get('source', f'page_{page_num}')
                
#                 metadata = {
#                     "source": source_info,
#                     "chunk_id": i,
#                     "page": page_num,
#                     "total_chunks": len(chunks),
#                     "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
#                 }
#                 metadatas.append(metadata)
                
#             logger.info(f"Sample metadata: {metadatas[0] if metadatas else 'None'}")
            
#             # Generate embeddings in batches to manage memory
#             logger.info("Generating embeddings...")
#             batch_size = 100
#             all_embeddings = []
            
#             for i in range(0, len(texts), batch_size):
#                 batch_texts = texts[i:i + batch_size]
#                 batch_embeddings = self.embeddings.embed_documents(batch_texts)
#                 all_embeddings.extend(batch_embeddings)
#                 logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
#             embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
#             # Create optimized FAISS index
#             faiss_index = self._create_faiss_index(embeddings_array)
            
#             # Create LangChain FAISS vector store using the simpler method
#             vectorstore = LangChainFAISS.from_texts(
#                 texts=texts,
#                 embedding=self.embeddings,
#                 metadatas=metadatas
#             )
            
#             # Replace the underlying index with our optimized one if we have many documents
#             if len(texts) >= 1000:
#                 # Only replace with custom index for large documents
#                 vectorstore.index = faiss_index

#             # Manage memory before storing new index
#             self._manage_memory()
            
#             # Store in memory and persist to disk
#             self.indices[chat_id] = vectorstore
#             vectorstore.save_local(str(self.index_dir / chat_id))
            
#             logger.info(f"Successfully processed PDF. Chat ID: {chat_id}")
#             return {
#                 "chat_id": chat_id,
#                 "chunks_created": len(chunks),
#                 "total_characters": total_chars,
#                 "index_type": type(faiss_index).__name__
#             }

#         except Exception as e:
#             logger.error(f"Error processing PDF: {str(e)}")
#             raise
#         finally:
#             # Clean up temporary file
#             if os.path.exists(temp_path):
#                 os.unlink(temp_path)

#     def ask_question(self, chat_id: str, query: str, k: int = 5) -> Dict[str, str]:
#         """Ask question with improved retrieval and response generation"""
#         try:
#             # Load vector store if not in memory
#             if chat_id not in self.indices:
#                 index_path = self.index_dir / chat_id
#                 if not index_path.exists():
#                     raise ValueError(f"No index found for chat_id: {chat_id}")
                
#                 logger.info(f"Loading index from disk for chat_id: {chat_id}")
#                 self.indices[chat_id] = LangChainFAISS.load_local(
#                     str(index_path),
#                     self.embeddings,
#                     allow_dangerous_deserialization=True
#                 )
#                 self._manage_memory()

#             # Retrieve vector store
#             vectorstore = self.indices[chat_id]
            
#             # Debug: Check if vectorstore has documents
#             logger.info(f"Vector store has {vectorstore.index.ntotal} documents")

#             # Use simple similarity search - most reliable
#             retriever = vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={"k": k}
#             )
            
#             # Test retrieval directly
#             docs = retriever.get_relevant_documents(query)
#             logger.info(f"Retrieved {len(docs)} documents for query: '{query[:50]}...'")
            
#             # Log retrieved content for debugging
#             for i, doc in enumerate(docs):
#                 logger.info(f"Doc {i}: {doc.page_content[:100]}...")
            
#             if not docs:
#                 logger.warning("No documents retrieved! Trying with all documents...")
#                 # Fallback: get all documents if none retrieved
#                 all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
#                 if all_docs:
#                     docs = all_docs[:k]
#                     logger.info(f"Using fallback: retrieved {len(docs)} documents")

#             # Set up QA chain - use the retriever that worked
#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=self.gemini,
#                 chain_type="stuff",
#                 retriever=retriever,
#                 return_source_documents=True,
#                 chain_type_kwargs={"prompt": self.custom_prompt}
#             )

#             # Run query
#             logger.info(f"Processing query with QA chain: {query[:100]}...")
#             result = qa_chain.invoke({"query": query})
            
#             logger.info(f"QA result keys: {result.keys()}")
#             logger.info(f"Response: {result.get('result', 'No result')[:200]}...")

#             # Extract source information
#             sources = []
#             if "source_documents" in result and result["source_documents"]:
#                 for doc in result["source_documents"]:
#                     sources.append({
#                         "page": doc.metadata.get("page", "unknown"),
#                         "source": doc.metadata.get("source", "unknown"),
#                         "chunk_id": doc.metadata.get("chunk_id", "unknown")
#                     })
#                 logger.info(f"Found {len(sources)} source documents")
#             else:
#                 logger.warning("No source documents in result")

#             return {
#                 "response": result.get("result", "No response generated"),
#                 "sources": sources,
#                 "num_sources": len(sources)
#             }

#         except Exception as e:
#             logger.error(f"Error answering question: {str(e)}", exc_info=True)
#             return {
#                 "response": f"Sorry, I encountered an error while processing your question: {str(e)}",
#                 "sources": [],
#                 "num_sources": 0
#             }

#     def get_chat_info(self, chat_id: str) -> Dict[str, any]:
#         """Get information about a chat session"""
#         if chat_id not in self.indices:
#             index_path = self.index_dir / chat_id
#             if not index_path.exists():
#                 raise ValueError(f"No index found for chat_id: {chat_id}")
        
#         # This is a simplified version - you might want to store more metadata
#         return {
#             "chat_id": chat_id,
#             "in_memory": chat_id in self.indices,
#             "index_path_exists": (self.index_dir / chat_id).exists()
#         }

#     def delete_index(self, chat_id: str):
#         """Delete index from memory and disk"""
#         # Remove from memory
#         if chat_id in self.indices:
#             del self.indices[chat_id]
#             logger.info(f"Removed index {chat_id} from memory")
        
#         # Remove from disk
#         index_path = self.index_dir / chat_id
#         if index_path.exists():
#             for file in index_path.glob("*"):
#                 file.unlink()
#             index_path.rmdir()
#             logger.info(f"Removed index {chat_id} from disk")
        
#         # Force garbage collection
#         gc.collect()

#     def list_active_chats(self) -> List[str]:
#         """List all available chat sessions"""
#         disk_chats = [d.name for d in self.index_dir.iterdir() if d.is_dir()]
#         memory_chats = list(self.indices.keys())
        
#         # Return unique chat IDs
#         return list(set(disk_chats + memory_chats))











import tempfile
import uuid
import os
import faiss
import numpy as np
import gc
import atexit
import signal
import sys
from typing import Dict, List, Tuple, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ChatProcessor:
    def __init__(self):
        # Load environment variables
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        
        # Load embedding model with optimized settings
        model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Initialize LLM with optimized settings
        self.gemini = GoogleGenerativeAI(
            model=gemini_model, 
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Store everything in RAM only - no persistent storage
        self.indices = {}
        self.chat_metadata = {}  # Store additional metadata for each chat
        
        # Memory management settings
        self.max_indices_in_memory = 10  # Increased since we're RAM-only
        self.max_memory_mb = 512  # Maximum memory usage in MB
        
        # Custom prompt template for better responses
        self.custom_prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Try to provide comprehensive answers based on the context provided.

            Context:
            {context}

            Question: {question}
            
            Answer:""",
            input_variables=["context", "question"]
        )
        
        # Register cleanup handlers for when server disconnects/shuts down
        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self):
        """Register handlers to clean up memory when server disconnects"""
        def cleanup_handler(signum=None, frame=None):
            logger.info("Server disconnecting - cleaning up all RAM data...")
            self.cleanup_all()
            if signum:
                logger.info(f"Received signal {signum}, exiting...")
                sys.exit(0)
        
        # Register for various exit scenarios
        atexit.register(cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)  # Termination signal
        signal.signal(signal.SIGINT, cleanup_handler)   # Ctrl+C
        
        # For Windows compatibility
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, cleanup_handler)

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB"""
        total_size = 0
        for chat_id, vectorstore in self.indices.items():
            # Rough estimation based on number of vectors and dimension
            if hasattr(vectorstore.index, 'ntotal'):
                num_vectors = vectorstore.index.ntotal
                # Assume 384 dimensions (typical for all-MiniLM-L6-v2) * 4 bytes per float
                vector_size = num_vectors * 384 * 4
                total_size += vector_size
        
        return total_size / (1024 * 1024)  # Convert to MB

    def _manage_memory(self):
        """Enhanced memory management for RAM-only storage"""
        current_memory = self._estimate_memory_usage()
        
        # Remove indices if we exceed memory limits
        while (len(self.indices) > self.max_indices_in_memory or 
               current_memory > self.max_memory_mb) and self.indices:
            
            # Remove oldest entry (simple FIFO strategy)
            oldest_chat_id = next(iter(self.indices))
            logger.info(f"Removing chat {oldest_chat_id} from memory (Memory: {current_memory:.1f}MB)")
            
            del self.indices[oldest_chat_id]
            if oldest_chat_id in self.chat_metadata:
                del self.chat_metadata[oldest_chat_id]
            
            gc.collect()  # Force garbage collection
            current_memory = self._estimate_memory_usage()

    def _get_optimal_chunk_size(self, total_chars: int) -> Tuple[int, int]:
        """Determine optimal chunk size based on document size"""
        if total_chars < 10000:  # Small documents
            return 800, 200
        elif total_chars < 50000:  # Medium documents
            return 1000, 250
        elif total_chars < 200000:  # Large documents
            return 1200, 300
        else:  # Very large documents
            return 1500, 400

    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Create optimized FAISS index for RAM storage"""
        n_embeddings, dimension = embeddings.shape
        
        if n_embeddings < 1000:
            # Use simple flat index for small datasets
            index = faiss.IndexFlatL2(dimension)
        elif n_embeddings < 10000:
            # Use IVF index for medium datasets
            nlist = min(int(np.sqrt(n_embeddings)), 100)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.nprobe = min(10, nlist)
        else:
            # Use IVF-PQ for large datasets to save memory
            nlist = min(int(np.sqrt(n_embeddings)), 1000)
            quantizer = faiss.IndexFlatL2(dimension)
            # Use 8-bit quantization with 8 sub-quantizers for memory efficiency
            index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
            index.nprobe = min(20, nlist)
        
        # Train index if needed
        if hasattr(index, 'train'):
            logger.info(f"Training FAISS index with {n_embeddings} vectors...")
            index.train(embeddings)
        
        # Add vectors to index
        index.add(embeddings)
        logger.info(f"Created FAISS index: {type(index).__name__} with {n_embeddings} vectors")
        
        return index

    def process_pdf(self, file_bytes: bytes, chunk_overlap_ratio: float = 0.2) -> Dict[str, str]:
        """Process PDF with RAM-only storage"""
        # Save PDF to temporary file (will be deleted after processing)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_path = temp_pdf.name

        try:
            logger.info("Loading PDF document...")
            # Load PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content found in PDF")
            
            # Calculate total characters for optimization
            total_chars = sum(len(doc.page_content) for doc in documents)
            logger.info(f"PDF loaded: {len(documents)} pages, {total_chars} total characters")
            
            # Get optimal chunk size
            chunk_size, chunk_overlap = self._get_optimal_chunk_size(total_chars)
            logger.info(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            
            # Split documents with optimized parameters
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")

            # Create unique chat ID
            chat_id = str(uuid.uuid4())

            # Prepare texts and enhanced metadata
            texts = [doc.page_content for doc in chunks]
            metadatas = []
            for i, doc in enumerate(chunks):
                # Extract page number from document metadata
                page_num = doc.metadata.get('page', 0)
                source_info = doc.metadata.get('source', f'page_{page_num}')
                
                metadata = {
                    "source": source_info,
                    "chunk_id": i,
                    "page": page_num,
                    "total_chunks": len(chunks),
                    "text": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                metadatas.append(metadata)
                
            logger.info(f"Sample metadata: {metadatas[0] if metadatas else 'None'}")
            
            # Generate embeddings in batches to manage memory
            logger.info("Generating embeddings...")
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            # Create LangChain FAISS vector store (RAM only)
            vectorstore = LangChainFAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # For large documents, replace with optimized index
            if len(texts) >= 1000:
                faiss_index = self._create_faiss_index(embeddings_array)
                vectorstore.index = faiss_index

            # Manage memory before storing new index
            self._manage_memory()
            
            # Store ONLY in RAM - no disk persistence
            self.indices[chat_id] = vectorstore
            self.chat_metadata[chat_id] = {
                "chunks_created": len(chunks),
                "total_characters": total_chars,
                "created_at": np.datetime64('now'),
                "document_name": "uploaded_pdf"
            }
            
            memory_usage = self._estimate_memory_usage()
            logger.info(f"Successfully processed PDF in RAM. Chat ID: {chat_id}, Memory usage: {memory_usage:.1f}MB")
            
            return {
                "chat_id": chat_id,
                "chunks_created": len(chunks),
                "total_characters": total_chars,
                "memory_usage_mb": memory_usage,
                "storage_type": "RAM_ONLY"
            }

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
        finally:
            # Always clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def ask_question(self, chat_id: str, query: str, k: int = 5) -> Dict[str, str]:
        """Ask question using RAM-only storage"""
        try:
            # Check if chat exists in RAM
            if chat_id not in self.indices:
                raise ValueError(f"Chat session {chat_id} not found in memory. It may have been cleaned up.")

            # Retrieve vector store from RAM
            vectorstore = self.indices[chat_id]
            
            # Debug: Check if vectorstore has documents
            logger.info(f"Vector store has {vectorstore.index.ntotal} documents")

            # Use simple similarity search
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            # Test retrieval directly
            docs = retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} documents for query: '{query[:50]}...'")
            
            # Log retrieved content for debugging
            for i, doc in enumerate(docs):
                logger.info(f"Doc {i}: {doc.page_content[:100]}...")
            
            if not docs:
                logger.warning("No documents retrieved! Trying with all documents...")
                # Fallback: get all documents if none retrieved
                all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
                if all_docs:
                    docs = all_docs[:k]
                    logger.info(f"Using fallback: retrieved {len(docs)} documents")

            # Set up QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.gemini,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.custom_prompt}
            )

            # Run query
            logger.info(f"Processing query with QA chain: {query[:100]}...")
            result = qa_chain.invoke({"query": query})
            
            logger.info(f"QA result keys: {result.keys()}")
            logger.info(f"Response: {result.get('result', 'No result')[:200]}...")

            # Extract source information
            sources = []
            if "source_documents" in result and result["source_documents"]:
                for doc in result["source_documents"]:
                    sources.append({
                        "page": doc.metadata.get("page", "unknown"),
                        "source": doc.metadata.get("source", "unknown"),
                        "chunk_id": doc.metadata.get("chunk_id", "unknown")
                    })
                logger.info(f"Found {len(sources)} source documents")
            else:
                logger.warning("No source documents in result")

            return {
                "response": result.get("result", "No response generated"),
                "sources": sources,
                "num_sources": len(sources),
                "storage_type": "RAM_ONLY"
            }

        except Exception as e:
            logger.error(f"Error answering question: {str(e)}", exc_info=True)
            return {
                "response": f"Sorry, I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "num_sources": 0,
                "storage_type": "RAM_ONLY"
            }

    def get_chat_info(self, chat_id: str) -> Dict[str, any]:
        """Get information about a RAM-stored chat session"""
        if chat_id not in self.indices:
            raise ValueError(f"Chat session {chat_id} not found in memory")
        
        metadata = self.chat_metadata.get(chat_id, {})
        memory_usage = self._estimate_memory_usage()
        
        return {
            "chat_id": chat_id,
            "storage_type": "RAM_ONLY",
            "in_memory": True,
            "total_memory_usage_mb": memory_usage,
            "chunks_created": metadata.get("chunks_created", "unknown"),
            "total_characters": metadata.get("total_characters", "unknown"),
            "created_at": str(metadata.get("created_at", "unknown"))
        }

    def delete_chat(self, chat_id: str):
        """Delete specific chat from RAM"""
        if chat_id in self.indices:
            del self.indices[chat_id]
            logger.info(f"Removed chat {chat_id} from RAM")
        
        if chat_id in self.chat_metadata:
            del self.chat_metadata[chat_id]
        
        # Force garbage collection
        gc.collect()
        
        memory_usage = self._estimate_memory_usage()
        logger.info(f"Memory usage after deletion: {memory_usage:.1f}MB")

    def list_active_chats(self) -> List[Dict[str, any]]:
        """List all active chat sessions in RAM"""
        chats = []
        for chat_id in self.indices.keys():
            metadata = self.chat_metadata.get(chat_id, {})
            chats.append({
                "chat_id": chat_id,
                "chunks_created": metadata.get("chunks_created", "unknown"),
                "total_characters": metadata.get("total_characters", "unknown"),
                "created_at": str(metadata.get("created_at", "unknown")),
                "storage_type": "RAM_ONLY"
            })
        
        return chats

    def get_memory_status(self) -> Dict[str, any]:
        """Get current memory usage status"""
        memory_usage = self._estimate_memory_usage()
        return {
            "total_chats": len(self.indices),
            "memory_usage_mb": memory_usage,
            "max_memory_mb": self.max_memory_mb,
            "max_chats": self.max_indices_in_memory,
            "memory_usage_percent": (memory_usage / self.max_memory_mb) * 100
        }

    def cleanup_all(self):
        """Clean up all data from RAM - called on server disconnect"""
        logger.info(f"Cleaning up {len(self.indices)} chat sessions from RAM...")
        
        self.indices.clear()
        self.chat_metadata.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("All RAM data cleaned up successfully")

# Global instance for cleanup
_processor_instance = None

def get_processor() -> ChatProcessor:
    """Get or create the global processor instance"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = ChatProcessor()
    return _processor_instance