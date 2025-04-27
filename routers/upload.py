
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import uuid
import os
import logging
from typing import List
from pydantic import BaseModel

from services.pdf_parser import PDFProcessor

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

class DocumentList(BaseModel):
    documents: List[DocumentResponse]

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
   
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    
    document_id = str(uuid.uuid4())
    
  
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the PDF in the background
        pdf_processor = PDFProcessor()
        background_tasks.add_task(
            pdf_processor.process_document,
            file_path,
            document_id
        )
        
        return DocumentResponse(
            document_id=document_id,
            filename=file.filename,
            status="processing"
        )
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


    


