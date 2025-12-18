import os
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import aiofiles
from pathlib import Path
import uuid

from ....models.schemas import FileUploadResponse, DocumentMetadata
from ....services.document_processor import DocumentProcessor
from ....services.vector_store import VectorStoreService
from ....core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
document_processor = DocumentProcessor()
vector_store_service = VectorStoreService()

async def save_upload_file(upload_file: UploadFile, destination: Path) -> str:
    """Save an uploaded file to the specified destination."""
    try:
        # Create a unique filename to prevent overwriting
        file_extension = Path(upload_file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = destination / unique_filename
        
        # Save the file asynchronously
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
            
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving file {upload_file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

@router.post("/upload", response_model=List[FileUploadResponse])
async def upload_documents(
    files: List[UploadFile] = File(...),
    metadata: Optional[str] = None
):
    """
    Upload and process documents for the knowledge base.
    
    This endpoint accepts multiple file uploads, saves them to the server,
    processes the content, and adds them to the vector store for retrieval.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    upload_responses = []
    processed_documents = []
    
    # Parse metadata if provided
    file_metadata = {}
    if metadata:
        try:
            file_metadata = DocumentMetadata.parse_raw(metadata).dict(exclude_unset=True)
        except Exception as e:
            logger.warning(f"Failed to parse metadata: {str(e)}")
    
    # Process each uploaded file
    for file in files:
        try:
            # Validate file type
            file_extension = Path(file.filename).suffix.lower()
            if file_extension[1:] not in settings.ALLOWED_FILE_TYPES:
                upload_responses.append({
                    "file_name": file.filename,
                    "message": f"File type {file_extension} not allowed",
                    "processed": False
                })
                continue
            
            # Save the uploaded file
            file_path = await save_upload_file(
                file,
                Path(settings.UPLOAD_FOLDER)
            )
            
            # Process the document
            try:
                # Add file to vector store
                documents = document_processor.process_files(
                    [file_path],
                    metadata={
                        **file_metadata,
                        "original_filename": file.filename
                    }
                )
                
                if documents:
                    vector_store_service.add_documents(documents)
                    processed_documents.extend(documents)
                    
                    upload_responses.append({
                        "file_name": file.filename,
                        "file_path": file_path,
                        "file_size": os.path.getsize(file_path),
                        "message": "File processed successfully",
                        "processed": True
                    })
                else:
                    upload_responses.append({
                        "file_name": file.filename,
                        "message": "No content could be extracted from the file",
                        "processed": False
                    })
                    
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                upload_responses.append({
                    "file_name": file.filename,
                    "message": f"Error processing file: {str(e)}",
                    "processed": False
                })
                
        except Exception as e:
            logger.error(f"Unexpected error with file {file.filename}: {str(e)}")
            upload_responses.append({
                "file_name": file.filename,
                "message": f"Unexpected error: {str(e)}",
                "processed": False
            })
    
    return upload_responses

@router.get("/documents")
async def list_documents():
    """List all documents in the knowledge base."""
    # This is a placeholder. In a real implementation, you would query your database
    # or vector store for the list of documents.
    return {"message": "List of documents will be returned here"}

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base."""
    # This is a placeholder. In a real implementation, you would remove the document
    # from your vector store and storage.
    return {"message": f"Document {document_id} will be deleted"}
