
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
import logging
import os

from services.langchain_service import RAGService, get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/qa",
    tags=["Question Answering"],
)

class QuestionRequest(BaseModel):
    document_id: str
    question: str
    top_k: int = 2  

class SourceChunk(BaseModel):
    text: str
    source: str  

class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    document_id: str
    question: str



@router.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
   
    document_id = request.document_id
    
    try:
        rag_service = get_rag_service()
        
        # Process the question
        answer, sources = rag_service.process_question(
            document_id=document_id,
            question=request.question,
            top_k=request.top_k 
        )
        
        # Format the sources
        formatted_sources = [
            SourceChunk(text=source.page_content, source=f"Page {source.metadata.get('page', 'unknown')}")
            for source in sources
        ]
        
        return QuestionResponse(
            answer=answer,
            sources=formatted_sources,
            document_id=document_id,
            question=request.question
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")
  
    document_id = request.document_id
    
 
    document_path = os.path.join(os.getcwd(), "uploads", f"{document_id}.pdf")
    if not os.path.exists(document_path):
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
       
        rag_service = RAGService()
        
        answer, sources = rag_service.process_question(
            document_id=document_id,
            question=request.question,
            top_k=request.top_k 
        )
        
        # Format the sources
        formatted_sources = [
            SourceChunk(text=source.page_content, source=f"Page {source.metadata.get('page', 'unknown')}")
            for source in sources
        ]
        
        return QuestionResponse(
            answer=answer,
            sources=formatted_sources,
            document_id=document_id,
            question=request.question
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")
