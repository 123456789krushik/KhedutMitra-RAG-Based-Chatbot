"""
rag_pipeline.py
---------------
Encapsulates the full RAG pipeline:
  1. Load FAISS index + embedding model (once at startup)
  2. clean_query()       → normalise user input
  3. retrieve()          → top-K similarity search
  4. build_prompt()      → inject context into strict prompt template
  5. generate_answer()   → call LLM (OpenAI or local fallback)
  6. ask()               → public entry point returning answer + sources
"""

# rag_pipeline.py — top section, replace existing imports and api_key lines

import re
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv   # ← ADD THIS

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ollama import chat
import os


client = None

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

INDEX_PATH      = Path("faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K           = 3
MIN_SCORE       = 0.10   # cosine similarity threshold (0-1); below = "no data"

# ─── Prompt Template (STRICT) ─────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are an expert agricultural advisor helping Indian farmers.
Answer ONLY using the provided context below. Do not use any outside knowledge.

Context:
{context}

Question: {question}

Instructions:
- Give practical, actionable agricultural advice.
- Use bullet points and numbered steps where helpful.
- Use simple language a farmer can understand.
- Include specific quantities, timings, or product names if present in the context.
- Include specific quantities, timings, or product names if present in the context.
- Answer confidently using whatever information is available in the context.
- Do NOT add any disclaimer, note, or "insufficient data" message at the end.
- Do NOT say "the context does not provide" or "please consult" at the end.
- If the context has ZERO relevant information, only then say:
  "I don't have specific information about this topic. Please ask your local KVK officer."
- NEVER mix a good answer with an insufficient data message together.

Answer:"""


# ─── Singleton loader ─────────────────────────────────────────────────────────

_vectorstore: Optional[FAISS] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None


def _load_vectorstore() -> FAISS:
    """Load (or return cached) FAISS index."""
    global _vectorstore, _embeddings

    if _vectorstore is not None:
        return _vectorstore

    if not INDEX_PATH.exists():
        raise RuntimeError(
            f"FAISS index not found at '{INDEX_PATH}'. "
            "Please run: python ingest.py"
        )

    logger.info("Loading embedding model…")
    _embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Loading FAISS index…")
    _vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        _embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("RAG pipeline ready.")
    return _vectorstore


# ─── Pipeline Steps ───────────────────────────────────────────────────────────

CROP_CORRECTIONS = {
    "promoganate": "pomegranate", "promogranate": "pomegranate",
    "pomagranate": "pomegranate", "pomogranate":  "pomegranate",
    "anar": "pomegranate", "anaar": "pomegranate",
    "aam": "mango", "keri": "mango",
    "kela": "banana", "kella": "banana",
    "gehun": "wheat", "gehu": "wheat",
    "chawal": "rice", "dhan": "rice",
    "makka": "maize", "angur": "grapes",
    "amrood": "guava", "papita": "papaya",
    "strawbery": "strawberry", "soya": "soybean",
}

def clean_query(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"[^\w\s\?\.\,\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [CROP_CORRECTIONS.get(w, w) for w in words]
    text = " ".join(words)
    for wrong, right in CROP_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text

def retrieve(query: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    store = _load_vectorstore()
    try:
        results = store.similarity_search_with_score(query, k=k)
        # Convert L2 distance to similarity (lower distance = higher similarity)
        converted = []
        for doc, score in results:
            # L2 distance: convert to 0-1 range
            similarity = 1 / (1 + score)
            converted.append((doc, similarity))
        # Filter by threshold
        filtered = [(doc, score) for doc, score in converted if score >= MIN_SCORE]
        logger.info(f"All scores: {[round(s,3) for _,s in converted]}")
        return filtered
    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        return []


def build_prompt(question: str, docs: list[tuple[Document, float]]) -> str:
    """
    Step 6 – Context Injection.
    Formats retrieved chunks into the strict prompt template.
    """
    if not docs:
        context = "No relevant context found."
    else:
        context_parts = []
        for i, (doc, score) in enumerate(docs, 1):
            meta = doc.metadata
            header = f"[Source {i} | crop: {meta.get('crop','?')} | topic: {meta.get('topic','?')} | relevance: {score:.2f}]"
            context_parts.append(f"{header}\n{doc.page_content}")
        context = "\n\n".join(context_parts)

    return PROMPT_TEMPLATE.format(context=context, question=question)


def _call_llama(prompt: str) -> str:
    response = chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def generate_answer(prompt: str, docs: list[tuple[Document, float]], history: list = []) -> str:
    try:
        return _call_llama(prompt)

    except Exception as e:
        logger.error(f"LLM call failed: {e}")

        # fallback to retrieved content
        if docs:
            return docs[0][0].page_content

        return "I couldn't generate an answer right now. Please try again later."
    
# ─── Public Entry Point ───────────────────────────────────────────────────────

# Store user's name across conversation
_user_name = None

GREET_PATTERN = re.compile(
    r"^(hi|hey|hello|hii|hiii|namaste|namaskar|salaam|"
    r"good\s*(morning|afternoon|evening|night)|"
    r"how are you|how r u|kaisa ho|"
    r"ok|okay|thanks|thank you|dhanyawad|bye|goodbye)[\s!\.]*$",
    re.IGNORECASE
)

NAME_PATTERN = re.compile(
    r"my\s+name\s+is\s+([A-Za-z]+)|mera\s+naam\s+([A-Za-z]+)|"
    r"i'm\s+([A-Za-z]+)|call\s+me\s+([A-Za-z]+)",
    re.IGNORECASE
)

# Words that should NOT be treated as names
NOT_A_NAME = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be",
    "to", "of", "in", "on", "at", "for", "with", "from", "by",
    "give", "going", "here", "just", "not", "also", "tell",
    "your", "my", "our", "their", "its", "this", "that",
    "krushik"  # remove this after testing
}

def _check_conversation(raw: str):
    global _user_name
    text = raw.strip()

    # Greeting check
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

    # "What is my name?" check
    if re.search(r"what\s+is\s+my\s+name|what'?s\s+my\s+name|my\s+name\s+kya|mera\s+naam\s+kya", text, re.IGNORECASE):
        if _user_name:
            return f"Your name is {_user_name}! 😊"
        else:
            return "You haven't told me your name yet! Please say 'My name is [your name]'."

    # Name introduction check
    match = NAME_PATTERN.search(text)
    if match:
        name = next((g for g in match.groups() if g), None)
        # Reject if it's a common word, not a real name
        if name and name.lower() not in NOT_A_NAME and len(name) > 2:
            name = name.capitalize()
            _user_name = name   # ← store name globally
            return (
                f"Namaste {name}! 🌾\n\n"
                f"Nice to meet you, {name}! I'm KhedutMitra.\n\n"
                f"Ask me anything about crops, fertilizers, pest control, "
                f"irrigation, or government schemes.\n\n"
                f"What would you like to know today, {name}?"
            )

    return None


def ask(raw_query: str, history: list = []) -> dict:
    # ── Conversation check — BEFORE RAG and Groq ──────────────────────────
    
    conv_answer = _check_conversation(raw_query)
    if conv_answer:                                       
        return {                                          
            "answer": conv_answer,                        
            "source_chunks": [],                          
            "clean_query": raw_query.strip().lower(),     
        }                                            
        
    """
    Full RAG pipeline.

    Args:
        raw_query: User's question (text or voice-transcribed text)

    Returns:
        {
            "answer":       str,
            "source_chunks": [{"content": str, "crop": str, "topic": str, "score": float}]
            "clean_query":  str,
        }
    """
    # Step 3
    query = clean_query(raw_query)
    logger.info(f"Clean query: {query!r}")

    # Steps 4 + 5
    docs = retrieve(query)
    logger.info(f"Retrieved {len(docs)} chunks above threshold")

    # Step 6
    prompt = build_prompt(query, docs)

    # Step 7
    # Step 7
    answer = generate_answer(prompt, docs, history)

    # Step 8 – format response
    source_chunks = [
        {
            "content": doc.page_content,
            "crop":    doc.metadata.get("crop", "general"),
            "topic":   doc.metadata.get("topic", "general"),
            "score": float(round(float(score), 3)),
        }
        for doc, score in docs
    ]

    return {
        "answer":        answer,
        "source_chunks": source_chunks,
        "clean_query":   query,
    }
