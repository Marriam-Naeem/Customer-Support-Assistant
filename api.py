import os
import json
import uuid
import logging
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our custom modules
from data_processor import DataProcessor
from vector_db_manager import VectorDatabaseManager
from llm_manager import LLMManager
from rag_guardrails import RAGPipeline, ConversationManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Banking Customer Service API",
    description="API for Banking Customer Service chatbot with RAG capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[Dict[str, Any]] = []
    rejected: bool = False
    rejection_reason: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    message: str

# Global variables
DATA_DIR = "data"
PROCESSED_DIR = "processed_data"
VECTOR_DIR = "vectorstore"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# Initialize components
data_processor = DataProcessor(output_dir=PROCESSED_DIR)
vector_manager = VectorDatabaseManager(persist_directory=VECTOR_DIR)
llm_manager = LLMManager(
    model_name="./models/phi_finetuned",  # Change to your preferred model
    model_type="causal",
    use_lora=False,
    quantize=False
)
rag_pipeline = RAGPipeline(vector_manager, llm_manager)
conversation_manager = ConversationManager()

# Dependency to get components
def get_components():
    return {
        "data_processor": data_processor,
        "vector_manager": vector_manager,
        "llm_manager": llm_manager,
        "rag_pipeline": rag_pipeline,
        "conversation_manager": conversation_manager
    }

# API endpoints
@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    components: Dict = Depends(get_components)
):
    """Process a user query and return a response."""
    try:
        logger.info(f"Received query: {request.query} from user {request.user_id}")
        
        # Get components
        rag_pipeline = components["rag_pipeline"]
        conversation_manager = components["conversation_manager"]
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Process query
        result = rag_pipeline.generate_response(request.query)
        
        # Add to conversation history
        conversation_manager.add_message(conversation_id, "user", request.query)
        conversation_manager.add_message(conversation_id, "assistant", result["response"])
        
        # Format sources
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "metadata": doc.metadata
            })
        
        return QueryResponse(
            response=result["response"],
            conversation_id=conversation_id,
            sources=sources,
            rejected=result.get("rejected", False),
            rejection_reason=result.get("rejection_reason")
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    components: Dict = Depends(get_components)
):
    """Upload and process a new document."""
    try:
        logger.info(f"Received document upload: {file.filename}")
        
        # Get components
        data_processor = components["data_processor"]
        vector_manager = components["vector_manager"]
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Save file
        file_path = os.path.join(DATA_DIR, f"{document_id}_{file.filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Process document in background
        background_tasks.add_task(
            process_document_task,
            file_path,
            document_id,
            data_processor,
            vector_manager
        )
        
        return DocumentUploadResponse(
            success=True,
            document_id=document_id,
            message=f"Document uploaded and being processed. Document ID: {document_id}"
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

async def process_document_task(
    file_path: str,
    document_id: str,
    data_processor: DataProcessor,
    vector_manager: VectorDatabaseManager
):
    """Process a document in the background."""
    try:
        logger.info(f"Processing document: {file_path}")
        
        # Process document
        processed_path = os.path.join(PROCESSED_DIR, f"{document_id}.json")
        num_chunks = data_processor.process_file(file_path)
        data_processor.save_processed_data(num_chunks, processed_path)
        
        # Add to vector store
        vector_manager.add_documents(processed_path)
        
        logger.info(f"Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}", exc_info=True)

@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    components: Dict = Depends(get_components)
):
    """Get conversation history for a specific conversation."""
    try:
        conversation_manager = components["conversation_manager"]
        history = conversation_manager.get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "history": history}
        
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    components: Dict = Depends(get_components)
):
    """Clear conversation history for a specific conversation."""
    try:
        conversation_manager = components["conversation_manager"]
        conversation_manager.clear_conversation(conversation_id)
        return {"message": f"Conversation {conversation_id} cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/stats")
async def stats(
    components: Dict = Depends(get_components)
):
    """Get system statistics."""
    try:
        vector_stats = components["vector_manager"].get_collection_stats()
        
        return {
            "vector_db": vector_stats,
            "model": {
                "name": components["llm_manager"].model_name,
                "type": components["llm_manager"].model_type
            }
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

# Run the app
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)