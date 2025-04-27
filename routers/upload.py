from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import os
import logging
from typing import List, Optional
from pydantic import BaseModel
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    page_count: int = 0
    cloudinary_url: Optional[str] = None

class DocumentList(BaseModel):
    documents: List[DocumentResponse]

class DocumentQuery(BaseModel):
    question: str

class DocumentAnswer(BaseModel):
    answer: str
    sources: List[dict]

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a PDF document, store it in Cloudinary, and process it for RAG"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate a unique document ID
    document_id = str(uuid.uuid4())
    
    # Temporary local file path
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
    
    try:
        # Save the file temporarily
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Upload to Cloudinary
        cloudinary_response = cloudinary.uploader.upload(
            file_path,
            resource_type="raw",
            public_id=f"pdf_documents/{document_id}",
            folder="rag_documents"
        )
        
        cloudinary_url = cloudinary_response.get('secure_url')
        
        # Import here to avoid circular imports
        from services.pdf_parser import PDFProcessor
        
        # Process the PDF in the background
        pdf_processor = PDFProcessor()
        background_tasks.add_task(
            pdf_processor.process_document_and_cleanup,
            file_path,
            document_id,
            cloudinary_url
        )
        
        return DocumentResponse(
            document_id=document_id,
            filename=file.filename,
            status="processing",
            cloudinary_url=cloudinary_url
        )
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        # Clean up temporary file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_status(document_id: str):
    """Get the status of a document"""
    try:
        from services.pdf_parser import PDFProcessor
        pdf_processor = PDFProcessor()
        status = pdf_processor.get_document_status(document_id)
        
        return DocumentResponse(
            document_id=document_id,
            filename=status.get("filename", "unknown"),
            status=status.get("status", "unknown"),
            page_count=status.get("page_count", 0),
            cloudinary_url=status.get("cloudinary_url")
        )
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated data"""
    try:
        from services.pdf_parser import PDFProcessor
        pdf_processor = PDFProcessor()
        
        # Get document status to retrieve cloudinary URL
        status = pdf_processor.get_document_status(document_id)
        
        # Delete from Cloudinary if available
        if status.get("cloudinary_url"):
            try:
                # Extract public_id from the URL
                public_id = f"rag_documents/pdf_documents/{document_id}"
                cloudinary.uploader.destroy(public_id, resource_type="raw")
            except Exception as e:
                logger.warning(f"Error deleting from Cloudinary: {str(e)}")
        
        # Delete local files and vector store
        pdf_processor.delete_document(document_id)
        
        return JSONResponse(content={"message": "Document deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@router.post("/{document_id}/query", response_model=DocumentAnswer)
async def query_document(document_id: str, query: DocumentQuery):
    """Query a document using RAG"""
    try:
        from services.langchain_service import get_rag_service
        
        rag_service = get_rag_service()
        answer, sources = rag_service.process_question(document_id, query.question)
        
        # Format sources for response
        formatted_sources = []
        for source in sources:
            formatted_sources.append({
                "content": source.page_content,
                "metadata": source.metadata
            })
        
        return DocumentAnswer(
            answer=answer,
            sources=formatted_sources
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found or not processed yet")
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/", response_model=DocumentList)
async def list_documents():
    """List all available documents"""
    try:
        from services.pdf_parser import PDFProcessor
        pdf_processor = PDFProcessor()
        documents = pdf_processor.list_documents()
        
        document_list = []
        for doc in documents:
            document_list.append(DocumentResponse(
                document_id=doc.get("document_id"),
                filename=doc.get("filename", "unknown"),
                status=doc.get("status", "unknown"),
                page_count=doc.get("page_count", 0),
                cloudinary_url=doc.get("cloudinary_url")
            ))
        
        return DocumentList(documents=document_list)
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")