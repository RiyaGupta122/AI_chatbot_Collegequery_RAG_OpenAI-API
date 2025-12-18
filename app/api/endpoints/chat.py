from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import os

from ....models.schemas import ChatRequest, ChatResponse, HealthCheck
from ....services.chat_service import ChatService
from ....services.vector_store import VectorStoreService
from ....core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
vector_store_service = VectorStoreService()
chat_service = ChatService(vector_store_service)

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {
        "status": "ok",
        "version": "1.0.0"
    }

@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    ""
    Handle chat messages and return AI responses.
    
    This endpoint processes user messages, maintains conversation context,
    and returns AI-generated responses based on the provided documents.
    """
    try:
        # Get the response from the chat service
        response = await chat_service.get_response(
            query=chat_request.message,
            chat_history=chat_request.chat_history
        )
        
        return ChatResponse(
            message=response["answer"],
            sources=response["sources"]
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )

@router.post("/reset-context")
async def reset_context():
    """
    Reset the conversation context.
    
    This clears the chat history and starts a new conversation.
    """
    try:
        chat_service.clear_memory()
        return {"message": "Conversation context has been reset."}
    except Exception as e:
        logger.error(f"Error resetting context: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while resetting the conversation context."
        )
