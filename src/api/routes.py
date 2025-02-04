from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import time
from datetime import datetime
import logging
from ..bot.chatbot import Chatbot

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Offline Chatbot API",
    description="API for document-based offline chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    context: Optional[Dict] = None
    include_debug_info: bool = False

class ChatResponse(BaseModel):
    response: str
    confidence: float
    source: str
    context: Optional[Dict]
    processing_time: float
    timestamp: str

def get_chatbot():
    if not hasattr(app.state, "chatbot"):
        app.state.chatbot = Chatbot("data/documents")
    return app.state.chatbot

@app.post("/chat", response_model=ChatResponse)
async def chat(
    query: Query,
    chatbot: Chatbot = Depends(get_chatbot)
) -> Dict:
    """
    Generate a response for the given query
    """
    start_time = time.time()
    
    try:
        response = chatbot.generate_response(query.text)
        
        response_data = {
            "response": response["response"],
            "confidence": response["confidence"],
            "source": response["source"],
            "context": response["context"] if query.include_debug_info else None,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Successfully processed query: {query.text[:50]}...")
        
        return response_data

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict:
    """
    Check API health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/stats")
async def get_stats(
    chatbot: Chatbot = Depends(get_chatbot)
) -> Dict:
    """
    Get chatbot statistics
    """
    return {
        "document_count": len(chatbot.documents),
        "conversation_history_length": len(chatbot.conversation_history),
        "supported_languages": chatbot.text_standardizer.nlp_models.keys()
    }

@app.post("/reload")
async def reload_documents(
    chatbot: Chatbot = Depends(get_chatbot)
) -> Dict:
    """
    Reload and reprocess all documents
    """
    try:
        chatbot.load_and_process_documents()
        return {
            "status": "success",
            "message": "Documents reloaded successfully",
            "document_count": len(chatbot.documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload documents: {str(e)}")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to track request processing time
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler
    """
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )
