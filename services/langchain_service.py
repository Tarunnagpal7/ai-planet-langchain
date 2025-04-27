
import logging
import json
from typing import Dict, List, Any, Tuple
import time
import traceback
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
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
        self.processed_dir = os.path.join(os.getcwd(), "processed")
        self.vector_dir = os.path.join(os.getcwd(), "vectorstores")
        
        os.makedirs(self.processed_dir, exist_ok=True)
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
        
        
        # google_api_key = "AIzaSyDiLFpADAFzHAqaA5NRSRvUPtvMKyIPq7Q"
        google_api_key = os.getenv("GOOGLE_API_KEY")
        print(f"Google API Key: {google_api_key}")
        # Set up LLM with explicit API key
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
            you can make conversation with the user, but you should not answer the question directly.
            If user ask something give answer context from pdf be conversative.

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
       
        vector_store_path = os.path.join(self.vector_dir, document_id)
        
     
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            return True
        return False
    
    def process_document_for_rag(self, document_id: str) -> bool:
        
        processed_file = os.path.join(self.processed_dir, f"{document_id}.json")
        
        if not os.path.exists(processed_file):
            logger.error(f"Processed document {document_id} not found")
            return False
        
        try:
            # Load the processed document
            with open(processed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            full_text = data.get("full_text", "")
            
            if not full_text.strip():
                logger.error(f"Document {document_id} has no text content")
                return False
                
            pages = data.get("pages", [])
            
         
            documents = []
            
            if not pages:
                
                doc = Document(
                    page_content=full_text,
                    metadata={"page": 0, "document_id": document_id}
                )
                documents.append(doc)
            else:
                for page in pages:
                    page_text = page.get("text", "").strip()
                    if page_text: 
                        doc = Document(
                            page_content=page_text,
                            metadata={"page": page.get("page_num", 0), "document_id": document_id}
                        )
                        documents.append(doc)
            
            
            if not documents:
                logger.error(f"No valid documents created for {document_id}")
                return False
                
            vector_store_path = os.path.join(self.vector_dir, document_id)
            
            if os.path.exists(vector_store_path):
                import shutil
                shutil.rmtree(vector_store_path)
                os.makedirs(vector_store_path, exist_ok=True)
            else:
                os.makedirs(vector_store_path, exist_ok=True)
            

            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Created {len(chunks)} chunks for document {document_id}")
            
            if not chunks:
                logger.error(f"No chunks created for document {document_id}")
                return False
            
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            vector_store.save_local(vector_store_path)
            
            logger.info(f"Vector store created for document {document_id}")
          
            if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
                logger.error(f"Vector store creation failed for {document_id}")
                return False
            
            metadata_file = os.path.join(self.processed_dir, f"{document_id}_metadata.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    
                    metadata["rag_processed"] = True
                    metadata["rag_processed_at"] = time.time()
                    metadata["chunks_count"] = len(chunks)
                    
                    with open(metadata_file, "w") as f:
                        json.dump(metadata, f)
                except Exception as e:
                    logger.error(f"Error updating metadata: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing document for RAG: {str(e)}")
            traceback.print_exc()
            return False
    
    def get_retriever(self, document_id: str, top_k: int = 2):
        """Get a retriever for a document"""
        vector_store_path = os.path.join(self.vector_dir, document_id)
        
        if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
            # Try to process the document
            success = self.process_document_for_rag(document_id)
            if not success:
                raise FileNotFoundError(f"Vector store for document {document_id} not found or creation failed")
        
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
            # If it fails, try rebuilding the vector store
            self.process_document_for_rag(document_id)
            # Try one more time with simpler approach
            vector_store = FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
            return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            
            vector_store_path = os.path.join(self.vector_dir, document_id)
            
            if not os.path.exists(vector_store_path) or not os.listdir(vector_store_path):
                # Try to process the document
                success = self.process_document_for_rag(document_id)
                if not success:
                    raise FileNotFoundError(f"Vector store for document {document_id} not found or creation failed")
            
            # Load the vector store
            vector_store = FAISS.load_local(vector_store_path, self.embeddings,allow_dangerous_deserialization=True)
            
            # Create retriever
            retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": top_k}
            )
            
            return retriever
    
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
   
        try:
            logger.info(f"Processing question for document {document_id}: {question}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if not self.is_document_processed(document_id):
             
                success = self.process_document_for_rag(document_id)
                if not success:
                    raise Exception("Document is not yet processed. Please wait and try again.")
            
         
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