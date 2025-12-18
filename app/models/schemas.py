from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str
    chat_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    message: str
    sources: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FileUploadResponse(BaseModel):
    message: str
    file_path: str
    file_name: str
    file_size: int
    processed: bool = False

class DocumentMetadata(BaseModel):
    source: str
    title: Optional[str] = None
    author: Optional[str] = None
    page: Optional[int] = None
    total_pages: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    custom_metadata: Dict[str, Any] = {}

class DocumentChunk(DocumentMetadata):
    id: str
    text: str
    embedding: Optional[List[float]] = None

class DocumentChunkWithScore(DocumentChunk):
    score: float

class QueryResult(BaseModel):
    query: str
    results: List[DocumentChunkWithScore] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthCheck(BaseModel):
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
