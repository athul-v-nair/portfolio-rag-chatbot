from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import time

from src.generation.generation import Generator

# Pydantic schemas
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    session_id: str = Field(..., description="Unique Session ID")
    top_k: int = Field(default=3, description="Number of top answers to retrieve")

# FastAPI Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifespan handler.

    This is executed once during application startup and shutdown.

    Startup:
        - Initialize shared resources (e.g., Generator instance)

    Shutdown:
        - Clean up resources if needed

    Args:
        app (FastAPI):
            The FastAPI application instance.

    Yields:
        None:
            Control is yielded back to the application runtime.
    """
    # Startup logic
    app.state.generator = Generator()

    yield

app = FastAPI(
    title="Portfolio RAG Chatbot",
    description="Ask anything about Athul V Nair.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow portfolio frontend origin 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Replace with your portfolio URL before deploying
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health", tags=["ops"])
async def health():
    """Liveness probe — hosting platforms ping this to keep the dyno alive."""
    return {"status": "ok", "timestamp": int(time.time())}

@app.post("/chat")
async def chat(request: ChatRequest, fastapi_request: Request):
    generator = fastapi_request.app.state.generator
    return StreamingResponse(
        generator.generate_answer(request.query, request.session_id, request.top_k),
        media_type="text/plain"
    )   