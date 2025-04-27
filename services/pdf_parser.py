# backend/services/pdf_parser.py
import fitz  # PyMuPDF
import os
import json
import logging
import uuid
from typing import Dict, List, Any
import time
import traceback

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
        self.processed_dir = os.path.join(os.getcwd(), "processed")
        
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def process_document(self, file_path: str, document_id: str) -> Dict[str, Any]:
       
        logger.info(f"Processing document: {document_id}")
        
        try:
            # Open the PDF file
            doc = fitz.open(file_path)
            
            # Create metadata
            metadata = {
                "document_id": document_id,
                "page_count": len(doc),
                "status": "processing",
                "processed_at": time.time(),
            }
            
            # Save initial metadata
            self._save_metadata(document_id, metadata)
            
            pages = []
            full_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                full_text += page_text
                
                pages.append({
                    "page_num": page_num,
                    "text": page_text,
                })
                
                logger.info(f"Extracted {len(page_text)} characters from page {page_num}")
            
            doc.close()
            

            logger.info(f"Total extracted text length: {len(full_text)} characters")
            
            processed_file = os.path.join(self.processed_dir, f"{document_id}.json")
            with open(processed_file, "w", encoding="utf-8") as f:
                json.dump({
                    "document_id": document_id,
                    "full_text": full_text,
                    "pages": pages,
                }, f, ensure_ascii=False)
            
            
            metadata["status"] = "processed"
            metadata["text_length"] = len(full_text)
            self._save_metadata(document_id, metadata)
            
            logger.info(f"Document {document_id} processed successfully")
            
            try:
                # Import here to avoid circular imports
                from services.langchain_service import RAGService
                rag_service = RAGService()
                rag_success = rag_service.process_document_for_rag(document_id)
                
                if rag_success:
                    metadata["rag_status"] = "processed"
                    self._save_metadata(document_id, metadata)
                    logger.info(f"RAG processing completed for document {document_id}")
                else:
                    metadata["rag_status"] = "failed"
                    self._save_metadata(document_id, metadata)
                    logger.error(f"RAG processing failed for document {document_id}")
            except Exception as e:
                logger.error(f"Error in automatic RAG processing: {str(e)}")
                metadata["rag_status"] = "error"
                metadata["rag_error"] = str(e)
                self._save_metadata(document_id, metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            traceback.print_exc()
            metadata = self.get_document_status(document_id)
            metadata["status"] = "failed"
            metadata["error"] = str(e)
            self._save_metadata(document_id, metadata)
            raise
    
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
     
        metadata_file = os.path.join(self.processed_dir, f"{document_id}_metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                return json.load(f)
        
        pdf_path = os.path.join(self.upload_dir, f"{document_id}.pdf")
        if os.path.exists(pdf_path):
            try:
                doc = fitz.open(pdf_path)
                metadata = {
                    "document_id": document_id,
                    "page_count": len(doc),
                    "status": "pending",
                }
                doc.close()
                return metadata
            except:
                return {
                    "document_id": document_id,
                    "status": "unknown",
                }
        
        return {
            "document_id": document_id,
            "status": "not_found",
        }
    
    def delete_document(self, document_id: str) -> bool:
        processed_file = os.path.join(self.processed_dir, f"{document_id}.json")
        metadata_file = os.path.join(self.processed_dir, f"{document_id}_metadata.json")
        vector_store_path = os.path.join(os.getcwd(), "vectorstores", document_id)
        
        # Delete processed files if they exist
        for file_path in [processed_file, metadata_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Delete vector store if it exists
        if os.path.exists(vector_store_path):
            import shutil
            shutil.rmtree(vector_store_path)
        
        return True
    
    def get_document_text(self, document_id: str) -> str:
        """Get the full text of a processed document"""
        processed_file = os.path.join(self.processed_dir, f"{document_id}.json")
        
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed document {document_id} not found")
        
        with open(processed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("full_text", "")
    
    def get_document_pages(self, document_id: str) -> List[Dict[str, Any]]:
        """Get the pages of a processed document"""
        processed_file = os.path.join(self.processed_dir, f"{document_id}.json")
        
        if not os.path.exists(processed_file):
            raise FileNotFoundError(f"Processed document {document_id} not found")
        
        with open(processed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("pages", [])
    
    def _save_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """Save metadata for a document"""
        metadata_file = os.path.join(self.processed_dir, f"{document_id}_metadata.json")
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)