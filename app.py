"""
app.py
------
FastAPI application.
Exposes:
  POST /ask          → main RAG endpoint
  GET  /health       → liveness check
  GET  /             → serves chat UI (index.html)
"""
 
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
 
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import rag_pipeline         
 
import os

# ─── Setup ────────────────────────────────────────────────────────────────────
 
load_dotenv()   # loads OPENAI_API_KEY from .env if present

api_key = os.getenv("OPENAI_API_KEY")
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
 
 
# ─── Lifespan: warm up the vector store at startup ────────────────────────────
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up RAG pipeline…")
    try:
        rag_pipeline._load_vectorstore()   # pre-loads model + index into memory
        logger.info("RAG pipeline ready.")
    except RuntimeError as e:
        logger.warning(f"⚠️  {e}")
        logger.warning("Server will start but /ask will fail until ingest.py is run.")
    yield
    logger.info("Shutting down.")
 
 
# ─── App ──────────────────────────────────────────────────────────────────────
 
app = FastAPI(
    title="Agri RAG Chatbot",
    description="Agricultural question answering via Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
)
 
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
 
 
# ─── Schemas ──────────────────────────────────────────────────────────────────
 
class AskRequest(BaseModel):
    query: str
 
    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query must not be empty.")
        if len(v) > 500:
            raise ValueError("Query too long (max 500 characters).")
        return v
 
 
class SourceChunk(BaseModel):
    content: str
    crop:    str
    topic:   str
    score:   float
 
 
class AskResponse(BaseModel):
    answer:        str
    source_chunks: list[SourceChunk]
    clean_query:   str
    latency_ms:    float
 
 
# ─── Routes ───────────────────────────────────────────────────────────────────
 
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the chat UI."""
    return templates.TemplateResponse(request=request, name="index.html")
 
 
@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok", "service": "agri-rag-chatbot"}
 
 
@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """
    Main RAG endpoint.

    Pipeline:
      1. Receive query (text/voice-transcribed)
      2. clean_query()
      3. retrieve() top-K chunks from FAISS
      4. build_prompt() with strict template
      5. generate_answer() via LLM
      6. Return answer + source chunks + latency
    """
    logger.info(f"Query: {body.query!r}")
    t0 = time.perf_counter()

    # ── Greeting Handler ──────────────────────────────────────────
    GREETINGS = ["how are you", "hello", "hi", "namaste", "hey", "good morning", "good evening"]
    if any(body.query.lower().strip().startswith(g) for g in GREETINGS):
        return AskResponse(
            answer="Namaste! 🌾 I'm your agricultural assistant. Ask me about crops, fertilizers, pest control, irrigation, or government schemes!",
            source_chunks=[],
            clean_query=body.query,
            latency_ms=0.0
        )
    # ── End Greeting Handler ──────────────────────────────────────

    try:
        result = rag_pipeline.ask(body.query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in RAG pipeline")
        raise HTTPException(status_code=500, detail="Internal server error.")

    latency = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"Answered in {latency} ms | chunks retrieved: {len(result['source_chunks'])}")

    return AskResponse(
        answer=result["answer"],
        source_chunks=result["source_chunks"],
        clean_query=result["clean_query"],
        latency_ms=latency,
    )