from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="College Information Chatbot")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
vector_store = None
qa_chain = None

class QueryRequest(BaseModel):
    question: str

class UploadResponse(BaseModel):
    message: str
    num_docs: int

def initialize_qa_chain(docs):
    global vector_store, qa_chain
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    texts = text_splitter.split_documents(docs)
    
    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # Initialize QA chain
    llm = OpenAI(temperature=0)  # Using OpenAI for better performance
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return len(texts)

@app.post("/upload/", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    all_docs = []
    
    for file in files:
        if file.filename.endswith('.pdf'):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Load PDF document
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                os.unlink(temp_file_path)
                raise HTTPException(status_code=400, detail=f"Error processing {file.filename}: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
    
    if not all_docs:
        raise HTTPException(status_code=400, detail="No valid PDF documents found")
    
    # Initialize QA chain with uploaded documents
    num_docs = initialize_qa_chain(all_docs)
    
    return {"message": "Documents processed successfully", "num_docs": num_docs}

@app.post("/query/")
async def query_documents(request: QueryRequest):
    if not qa_chain:
        raise HTTPException(status_code=400, detail="Please upload documents first")
    
    try:
        result = qa_chain({"query": request.question})
        
        # Format the response
        response = {
            "answer": result["result"],
            "sources": [doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]]
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "College Information Chatbot API is running. Use /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
