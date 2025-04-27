# backend/services/pdf_parser.py
import fitz  # PyMuPDF
import os
import json
import logging
import uuid
from typing import Dict, List, Any
import time
import traceback
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        self.upload_dir = os.path.join(os.getcwd(), "uploads")
        self.metadata_dir = os.path.join(os.getcwd(), "metadata")
        self.vector_dir = os.path.join(os.getcwd(), "vectorstores")
        
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.vector_dir, exist_ok=True)
    
    def process_document_and_cleanup(self, file_path: str, document_id: str, cloudinary_url: str = None) -> Dict[str, Any]:
        """Process document and remove local file after processing"""
        try:
            # Process the document
            metadata = self.process_document(file_path, document_id, cloudinary_url)
            
            # Delete the local PDF file after successful processing
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted local PDF file for document {document_id}")
            
            return metadata
        except Exception as e:
            logger.error(f"Error in process_document_and_cleanup: {str(e)}")
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise
    
    def process_document(self, file_path: str, document_id: str, cloudinary_url: str = None) -> Dict[str, Any]:
        """Process a document and extract text"""
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
                "filename": os.path.basename(file_path),
                "cloudinary_url": cloudinary_url
            }
            
            # Save initial metadata
            self._save_metadata(document_id, metadata)
            
            # Extract text directly for RAG processing
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                full_text += page_text
                logger.info(f"Extracted {len(page_text)} characters from page {page_num}")
            
            doc.close()
            
            logger.info(f"Total extracted text length: {len(full_text)} characters")
            
            # Update metadata
            metadata["status"] = "text_extracted"
            metadata["text_length"] = len(full_text)
            self._save_metadata(document_id, metadata)
            
            # Process for RAG directly without saving intermediate files
            try:
                from services.langchain_service import get_rag_service
                rag_service = get_rag_service()
                
                # Process the document text directly
                rag_success = rag_service.process_text_for_rag(document_id, full_text)
                
                if rag_success:
                    metadata["status"] = "processed"
                    metadata["rag_status"] = "processed"
                    self._save_metadata(document_id, metadata)
                    logger.info(f"RAG processing completed for document {document_id}")
                else:
                    metadata["status"] = "rag_failed"
                    metadata["rag_status"] = "failed"
                    self._save_metadata(document_id, metadata)
                    logger.error(f"RAG processing failed for document {document_id}")
            except Exception as e:
                logger.error(f"Error in RAG processing: {str(e)}")
                metadata["status"] = "rag_error"
                metadata["rag_status"] = "error"
                metadata["rag_error"] = str(e)
                self._save_metadata(document_id, metadata)
                raise
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            traceback.print_exc()
            metadata = self.get_document_status(document_id)
            metadata["status"] = "failed"
            metadata["error"] = str(e)
            self._save_metadata(document_id, metadata)
            raise
    
    def download_from_cloudinary(self, cloudinary_url: str) -> BytesIO:
        """Download a file from Cloudinary URL"""
        response = requests.get(cloudinary_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download file from Cloudinary: {response.status_code}")
        return BytesIO(response.content)
    
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """Get the status of a document"""
        metadata_file = os.path.join(self.metadata_dir, f"{document_id}.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                return json.load(f)
        
        # Check if document exists in uploads folder
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
        """Delete a document and all related files"""
        metadata_file = os.path.join(self.metadata_dir, f"{document_id}.json")
        vector_store_path = os.path.join(self.vector_dir, document_id)
        
        # Delete any local PDF file if it exists
        pdf_file = os.path.join(self.upload_dir, f"{document_id}.pdf")
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
        
        # Delete metadata if it exists
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        
        # Delete vector store if it exists
        if os.path.exists(vector_store_path):
            import shutil
            shutil.rmtree(vector_store_path)
        
        return True
    
    def _save_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """Save metadata for a document"""
        metadata_file = os.path.join(self.metadata_dir, f"{document_id}.json")
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the system"""
        documents = []
        
        # Look through metadata files
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith(".json"):
                document_id = filename.replace(".json", "")
                
                try:
                    with open(os.path.join(self.metadata_dir, filename), "r") as f:
                        metadata = json.load(f)
                        documents.append(metadata)
                except Exception as e:
                    logger.error(f"Error reading metadata file {filename}: {str(e)}")
        
        return documents