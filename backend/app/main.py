from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from rag_engine.vector_store import VectorStore
from rag_engine.rag_engine import MarketLensRAG

app = FastAPI(
    title="MarketLens API",
    description="RAG Backend for financial document analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4005"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Starting MarketLens Server")

try:
    vector_db = VectorStore()
    print("Vector Database connected successfully")

except Exception as e:
    print(f"CRITICAL WARNING: Vector Store connection failed: {e}")
    vector_db = None



class ChatRequest(BaseModel):
    question : str

class ChatResponse(BaseModel):
    answer: str
    sources_count: str



@app.get(
    "/api/health", 
    status_code=status.HTTP_200_OK,
    tags=["System Diagnostics"]
)
def health_check():
    """A simple endpoint to check if the server is alive."""
    return {"status": "MarketLens API is running normally."}

@app.post(
    "/api/chat", 
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Core AI Engine"],
    responses={
        200: {"description": "Successfully generated AI response."},
        401: {"description": "Unauthorized - Missing or Invalid Google API Key."},
        422: {"description": "Unprocessable Entity - Invalid JSON body format."},
        500: {"description": "Internal Server Error - Database disconnected or LLM failure."}
    }
)
def chat_endpoint(request: ChatRequest, x_api_key: str = Header(...)):
    """
    The main AI generation endpoint.
    - Requires 'x-api-key' in the HTTP header.
    - Requires JSON body: {"question": "What is the revenue?"}
    """
    if not vector_db:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Database is not initialized."
        )

    try:
        print(f"Received question: {request.question}")
        retrieved_texts = vector_db.search(query=request.question, n_results=3)

        rag_engine = MarketLensRAG(user_api_key=x_api_key)

        answer = rag_engine.generate_answer(query=request.question, retrieved_contexts=retrieved_texts)

        return ChatResponse(answer=answer, sources_count=len(retrieved_texts))

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail=str(ve)
        )
    except Exception as e:
        print(f"Server Error during chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An error occurred while generating the response."
        )
