"""
app.py
------
FastAPI application.
Exposes:
  POST /ask          → main RAG endpoint
  GET  /health       → liveness check
  GET  /             → serves chat UI (index.html)
"""
 
import re
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
 
load_dotenv()   

# Ensure the Hugging Face token is loaded for the pipeline
hf_token = os.getenv("HF_TOKEN")
 
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
 
class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []

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
 
 
# ─── Conversation Detection ───────────────────────────────────────────────────

GREET_PATTERN = re.compile(
    r"^(hi|hey|hello|hii|namaste|namaskar|salaam|"
    r"good\s*(morning|afternoon|evening|night)|"
    r"how are you|how r u|kaisa ho|"
    r"ok|okay|thanks|thank you|dhanyawad|bye|goodbye)[\s!\.]*$",
    re.IGNORECASE
)
NAME_PATTERN = re.compile(
    r"my\s+name\s+is\s+(\w+)|mera\s+naam\s+(\w+)|"
    r"i\s+am\s+(\w+)|i'm\s+(\w+)|call\s+me\s+(\w+)",
    re.IGNORECASE
)

def _check_conversation(raw: str):
    text = raw.strip()
    if GREET_PATTERN.match(text):
        return (
            "Namaste! 🌾 I'm KhedutMitra, your agricultural assistant.\n\n"
            "You can ask me about:\n"
            "• 🌱 Crop cultivation\n"
            "• 💊 Fertilizer recommendations\n"
            "• 🐛 Pest & disease management\n"
            "• 💧 Irrigation guidance\n"
            "• 🏛️ Government schemes\n\n"
            "How can I help your farm today?"
        )
    match = NAME_PATTERN.search(text)
    if match:
        name = next(g for g in match.groups() if g).capitalize()
        return (
            f"Namaste {name}! 🌾\n\n"
            f"Nice to meet you, {name}! I'm KhedutMitra.\n\n"
            f"Ask me anything about crops, fertilizers, pest control, "
            f"irrigation, or government schemes.\n\n"
            f"What would you like to know today, {name}?"
        )
    return None


@app.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    logger.info(f"Query: {body.query!r}")
    t0 = time.perf_counter()

    # ── Step 1: Conversation check (greetings / name) ─────────────────────
    conv_answer = _check_conversation(body.query)
    if conv_answer:
        return AskResponse(
            answer=conv_answer,
            source_chunks=[],
            clean_query=body.query.lower().strip(),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    # ── Step 2: Agricultural RAG with history ─────────────────────────────
    history = [{"role": m.role, "content": m.content} for m in body.history]

    try:
        result = rag_pipeline.ask(body.query, history)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in RAG pipeline")
        raise HTTPException(status_code=500, detail="Internal server error.")

    latency = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"Answered in {latency} ms | chunks: {len(result['source_chunks'])}")

    return AskResponse(
        answer=result["answer"],
        source_chunks=result["source_chunks"],
        clean_query=result["clean_query"],
        latency_ms=latency,
    )