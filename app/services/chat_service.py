import logging
from typing import List, Dict, Any, Optional
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from ..core.config import settings

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling chat interactions with the LLM."""
    
    def __init__(self, vector_store_service):
        self.vector_store_service = vector_store_service
        self.llm = self._initialize_llm()
        self.memory = self._initialize_memory()
        self.qa_chain = self._create_qa_chain()
        
    def _initialize_llm(self):
        """Initialize the language model."""
        return ChatOpenAI(
            model_name=settings.OPENAI_MODEL_NAME,
            temperature=0.7,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=True
        )
    
    def _initialize_memory(self):
        """Initialize conversation memory."""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the chatbot."""
        return """You are an AI-powered college information assistant. Your role is to answer questions about the college 
        using ONLY the information provided in the retrieved context. Be clear, concise, and helpful. If you don't know 
        the answer, say you don't know. Don't make up information.
        
        Guidelines:
        - Be polite and professional
        - Keep answers concise but complete
        - Use bullet points for lists when appropriate
        - If the answer is not in the context, say so
        - Always respond in the same language as the question
        """
    
    def _create_qa_chain(self):
        """Create the question-answering chain."""
        # Define the prompt template
        system_template = self._get_system_prompt()
        
        messages = [
            SystemMessage(content=system_template),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("""
            Context:
            {context}
            
            Question: {question}
            
            Answer in the same language as the question:""")
        ]
        
        qa_prompt = ChatPromptTemplate.from_messages(messages)
        
        # Create the QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store_service.get_retriever(
                search_kwargs={"k": 4}
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        return qa_chain
    
    async def get_response(self, query: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """Get a response for the given query."""
        if not query.strip():
            return {"answer": "Please provide a valid question.", "sources": []}
        
        try:
            # Prepare the chat history if provided
            if chat_history:
                self._update_chat_history(chat_history)
            
            # Get the response from the QA chain
            result = await self.qa_chain.acall({"question": query, "chat_history": chat_history or []})
            
            # Extract sources from the retrieved documents
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        source = doc.metadata["source"]
                        if source not in sources:
                            sources.append(source)
            
            return {
                "answer": result.get("answer", "I'm sorry, I couldn't find an answer to your question."),
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I'm sorry, I encountered an error while processing your request. Please try again later.",
                "sources": []
            }
    
    def _update_chat_history(self, chat_history: List[Dict]) -> None:
        """Update the conversation memory with chat history."""
        self.memory.clear()
        for message in chat_history:
            if message["role"] == "user":
                self.memory.chat_memory.add_user_message(message["content"])
            else:
                self.memory.chat_memory.add_ai_message(message["content"])
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
        logger.info("Chat memory cleared")
