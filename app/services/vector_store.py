import os
import logging
from typing import List, Optional, Dict, Any
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.docstore.document import Document as LangchainDocument
from ..core.config import settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Service for handling document embeddings and similarity search."""
    
    def __init__(self):
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.vector_store_path = settings.VECTOR_STORE_PATH
        self.index_file = os.path.join(self.vector_store_path, "faiss_index")
        
        # Create vector store directory if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Try to load existing vector store if it exists
        self._load_existing_vector_store()
    
    def _initialize_embeddings(self):
        """Initialize the appropriate embeddings model."""
        if settings.OPENAI_API_KEY:
            logger.info("Using OpenAI embeddings")
            return OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )
        else:
            logger.info("Using HuggingFace embeddings")
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def _load_existing_vector_store(self):
        """Load an existing vector store if it exists."""
        if os.path.exists(self.index_file):
            try:
                logger.info(f"Loading existing vector store from {self.index_file}")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True  # Required for FAISS
                )
                logger.info(f"Successfully loaded vector store with {self.vector_store.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Error loading existing vector store: {str(e)}")
                self.vector_store = None
    
    def add_documents(self, documents: List[LangchainDocument]) -> None:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        if self.vector_store is None:
            # Create a new vector store
            logger.info("Creating new vector store")
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            # Add to existing vector store
            logger.info("Adding to existing vector store")
            self.vector_store.add_documents(documents)
        
        # Save the updated vector store
        self._save_vector_store()
    
    def _save_vector_store(self) -> None:
        """Save the current vector store to disk."""
        if self.vector_store is not None:
            try:
                self.vector_store.save_local(self.vector_store_path)
                logger.info(f"Vector store saved to {self.vector_store_path}")
            except Exception as e:
                logger.error(f"Error saving vector store: {str(e)}")
                raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[LangchainDocument]:
        """Search for similar documents to the query."""
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []
        
        try:
            # If filter_dict is provided, use similarity_search_with_score
            if filter_dict:
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict,
                    **kwargs
                )
                # Extract just the documents (without scores)
                return [doc for doc, _ in docs_with_scores]
            else:
                return self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """Get a retriever for the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized")
        
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs or {"k": 4, "search_type": "similarity"}
        )
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.vector_store = None
        # Remove existing vector store files
        for file in os.listdir(self.vector_store_path):
            if file.startswith("faiss_index"):
                os.remove(os.path.join(self.vector_store_path, file))
        logger.info("Vector store cleared")
