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
from langchain.docstore.document import Document

from openai import OpenAI
import os

load_dotenv()

# Set HuggingFace token
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

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
- If the context does not contain enough information to answer, respond EXACTLY with:
  "Insufficient data: I could not find reliable information about this in my knowledge base. Please consult your local Krishi Vigyan Kendra (KVK) or agricultural extension officer."

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

def clean_query(raw: str) -> str:
    """
    Step 3 – Query Cleaning.
    Lowercase, strip excess whitespace, remove special characters
    that don't add semantic meaning.
    """
    text = raw.strip().lower()
    text = re.sub(r"[^\w\s\?\.\,\-]", " ", text)   # keep basic punctuation
    text = re.sub(r"\s+", " ", text)                 # collapse whitespace
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


def _call_openai(prompt: str) -> str:
    """Call Groq LLM (free alternative to OpenAI)."""
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def _call_fallback(prompt: str, docs: list[tuple[Document, float]]) -> str:
    """
    Fallback when no OpenAI key is set.
    Returns a structured answer directly from retrieved chunks.
    No hallucination possible — only retrieval content is used.
    """
    if not docs:
        return (
            "Insufficient data: I could not find reliable information about this "
            "in my knowledge base. Please consult your local Krishi Vigyan Kendra "
            "(KVK) or agricultural extension officer."
        )

    lines = ["Here is what I found in my knowledge base:\n"]
    for i, (doc, score) in enumerate(docs, 1):
        meta = doc.metadata
        lines.append(
            f"**Source {i}** (crop: {meta.get('crop','general')}, "
            f"topic: {meta.get('topic','general')}, score: {score:.2f}):\n"
            f"{doc.page_content}\n"
        )
    lines.append(
        "\n_Note: This answer is retrieved directly from the knowledge base. "
        "For detailed advice, set OPENAI_API_KEY to enable AI-generated responses._"
    )
    return "\n".join(lines)


def generate_answer(prompt: str, docs: list[tuple[Document, float]]) -> str:
    """
    Step 7 – LLM Response Generation.
    Uses Groq if key is available, otherwise structured fallback.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if api_key:
        try:
            return _call_openai(prompt)
        except Exception as e:
            logger.error(f"Groq call failed: {e}")
            return _call_fallback(prompt, docs)
    else:
        logger.warning("GROQ_API_KEY not set — using retrieval-only fallback.")
        return _call_fallback(prompt, docs)


# ─── Public Entry Point ───────────────────────────────────────────────────────

def ask(raw_query: str) -> dict:
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
    answer = generate_answer(prompt, docs)

    # Step 8 – format response
    source_chunks = [
        {
            "content": doc.page_content,
            "crop":    doc.metadata.get("crop", "general"),
            "topic":   doc.metadata.get("topic", "general"),
            "score":   round(score, 3),
        }
        for doc, score in docs
    ]

    return {
        "answer":        answer,
        "source_chunks": source_chunks,
        "clean_query":   query,
    }
