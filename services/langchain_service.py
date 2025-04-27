import logging
import json
from typing import Dict, List, Any, Tuple
import time
import traceback
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

_rag_service_instance = None
logger = logging.getLogger(__name__)

def get_rag_service():
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance

class RAGService:
    def __init__(self):
        self.vector_dir = os.path.join(os.getcwd(), "vectorstores")
        
        os.makedirs(self.vector_dir, exist_ok=True)
        
        # Initialize embeddings
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                model_kwargs={"device": device}
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            traceback.print_exc()
            raise
        
        # Set up LLM
        google_api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            temperature=0.2,
            google_api_key=google_api_key
        )
       
        # QA prompt template
        self.prompt = PromptTemplate(
            template="""
            You are a helpful assistant. Answer ONLY from the provided context. 
            If the context is insufficient, just say you don't know.
            You can make conversation with the user, but you should not answer the question directly.
            If user asks something, give answer context from pdf be conversative.

            Context:
            {context}

            Question: {question}
            
            Provide a detailed and accurate answer based solely on the given context.
            """,
            input_variables=["context", "question"]
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        
        self.parser = StrOutputParser()
    
    def is_document_processed(self, document_id: str) -> bool:
        """Check if vector store exists for document"""
        vector_store_path = os.path.join(self.vector_dir, document_id)
        
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            return True
        return False
    
    def process_text_for_rag(self, document_id: str, text: str) -> bool:
        """Process text directly for RAG without requiring processed files"""
        try:
            logger.info(f"Processing text for RAG: document_id={document_id}")
            
            if not text.strip():
                logger.error(f"Document {document_id} has no text content")
                return False
                
            # Create a document from text
            doc = Document(
                page_content=text,
                metadata={"document_id": document_id, "page": 0}
            )
            
            # Create vector store directory
            vector_store_path = os.path.join(self.vector_dir, document_id)
            if os.path.exists(vector_store_path):
                import shutil
                shutil.rmtree(vector_store_path)
            os.makedirs(vector_store_path, exist_ok=True)
            
            # Split text into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            logger.info(f"Created {len(chunks)} chunks for document {document_id}")
            
            if not chunks:
                logger.error(f"No chunks created for document {document_id}")
                return False
            
            # Create vector store
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(vector_store_path)
            
            logger.info(f"Vector store created for document {document_id}")
            
            # Verify vector store was created
            if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
                logger.error(f"Vector store creation failed for {document_id}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error processing text for RAG: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_retriever(self, document_id: str, top_k: int = 2):
        """Get a retriever for a document"""
        vector_store_path = os.path.join(self.vector_dir, document_id)
        
        if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
            raise FileNotFoundError(f"Vector store for document {document_id} not found")
        
        try:
            # Force CPU for FAISS loading if needed
            device = None
            if hasattr(self.embeddings, "client") and hasattr(self.embeddings.client, "device"):
                device = self.embeddings.client.device
                if device != "cpu":
                    self.embeddings.client = self.embeddings.client.to("cpu")
            
            # Load the vector store
            vector_store = FAISS.load_local(
                vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            if device is not None and device != "cpu":
                self.embeddings.client = self.embeddings.client.to(device)
            
            # Create retriever
            retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": top_k}
            )
            
            return retriever
        except Exception as e:
            logger.error(f"Error getting retriever: {str(e)}")
            raise
    
    def build_rag_chain(self, document_id: str, top_k: int = 2):
        """Build a RAG chain for a document"""
        retriever = self.get_retriever(document_id, top_k)
        
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        # Create main chain
        chain = parallel_chain | self.prompt | self.llm | self.parser
        
        return chain, retriever
    
    def process_question(self, document_id: str, question: str, top_k: int = 2) -> Tuple[str, List[Document]]:
        """Process a question for a document"""
        try:
            logger.info(f"Processing question for document {document_id}: {question}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if not self.is_document_processed(document_id):
                raise Exception("Document is not processed for RAG. Please try again later.")
            
            chain, retriever = self.build_rag_chain(document_id, top_k)
            
            # Get source documents
            sources = retriever.invoke(question)
            
            # Get answer
            answer = chain.invoke(question)
            
            # Clear cache again after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return answer, sources
                
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            traceback.print_exc()
            raise