import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ollama import chat

load_dotenv()

logger = logging.getLogger(__name__)

INDEX_PATH = Path("faiss_index")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
MIN_SCORE = 0.10

OLLAMA_MODEL = "gemma2:2b"
OLLAMA_TIMEOUT = 30

PROMPT_TEMPLATE = """\
You are an expert agricultural advisor helping Indian farmers.
Answer ONLY using the provided context below. Do not use any outside knowledge.

{conversation_history}
Context:
{context}

Question: {question}

Instructions:
- Give practical, actionable agricultural advice.
- Use bullet points and numbered steps where helpful.
- Use simple language a farmer can understand.
- Include specific quantities, timings, or product names if present in the context.
- Answer confidently using whatever information is available in the context.
- Do NOT add any disclaimer, note, or "insufficient data" message at the end.
- Do NOT say "the context does not provide" or "please consult" at the end.
- If the context has ZERO relevant information, only then say:
  "I don't have specific information about this topic. Please ask your local KVK officer."
- NEVER mix a good answer with an insufficient data message together.

Answer:"""

CROP_CORRECTIONS = {
    "promoganate": "pomegranate", "promogranate": "pomegranate",
    "pomagranate": "pomegranate", "pomogranate": "pomegranate",
    "anar": "pomegranate", "anaar": "pomegranate",
    "aam": "mango", "keri": "mango", "manga": "mango",
    "kela": "banana", "kella": "banana",
    "gehun": "wheat", "gehu": "wheat", "gahu": "wheat",
    "chawal": "rice", "dhan": "rice", "dhaan": "rice",
    "makka": "maize", "corn": "maize",
    "angur": "grapes", "angoor": "grapes",
    "amrood": "guava", "amrud": "guava",
    "papita": "papaya", "paw paw": "papaya",
    "strawbery": "strawberry", "strobery": "strawberry",
    "soya": "soybean", "soyabean": "soybean",
    "tur": "pigeon pea", "arhar": "pigeon pea",
    "urd": "black gram", "urad": "black gram",
    "moong": "mung bean", "mung": "mung bean",
}

GREET_PATTERN = re.compile(
    r"^(hi|hey|hello|hii|hiii|namaste|namaskar|salaam|"
    r"good\s*(morning|afternoon|evening|night)|"
    r"how are you|how r u|kaisa ho|kaise ho|"
    r"ok|okay|thanks|thank you|thankyou|dhanyawad|shukriya|"
    r"bye|goodbye|see you|see ya)[\s!\.]*$",
    re.IGNORECASE
)

NAME_PATTERN = re.compile(
    r"my\s+name\s+is\s+([A-Za-z]+)|mera\s+naam\s+([A-Za-z]+)|"
    r"i'm\s+([A-Za-z]+)|im\s+([A-Za-z]+)|call\s+me\s+([A-Za-z]+)|"
    r"mujhe\s+([A-Za-z]+)\s+kehte\s+hain|naam\s+([A-Za-z]+)\s+hai",
    re.IGNORECASE
)

NOT_A_NAME = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be",
    "to", "of", "in", "on", "at", "for", "with", "from", "by",
    "give", "going", "here", "just", "not", "also", "tell",
    "your", "my", "our", "their", "its", "this", "that",
    "krushik", "test", "user", "farmer", "sir", "ji",
}

_vectorstore: Optional[FAISS] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_user_name: Optional[str] = None
_conversation_history: List[Tuple[str, str]] = []


def _load_vectorstore() -> FAISS:
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


def _check_conversation(raw: str) -> Optional[str]:
    global _user_name
    text = raw.strip()

    if GREET_PATTERN.match(text):
        greeting = "Namaste! 🌾 I'm KhedutMitra, your agricultural assistant."
        if _user_name:
            greeting = f"Namaste {_user_name}! 🌾 Welcome back!"

        return (
            f"{greeting}\n\n"
            "I can help with:\n"
            "• 🌱 Crop cultivation & best practices\n"
            "• 💊 Fertilizer & nutrient management\n"
            "• 🐛 Pest & disease control\n"
            "• 💧 Irrigation & water management\n"
            "• 🏛️ Government schemes & subsidies\n"
            "• 📊 Yield improvement strategies\n\n"
            "What can I help with today?"
        )

    if re.search(
        r"what\s+is\s+my\s+name|what'?s\s+my\s+name|"
        r"my\s+name\s+kya|mera\s+naam\s+kya|"
        r"mujhe\s+kya\s+bolte\s+ho|you\s+know\s+my\s+name",
        text,
        re.IGNORECASE
    ):
        if _user_name:
            return f"Your name is {_user_name}! 😊"
        else:
            return (
                "You haven't told me your name yet! Please say:\n"
                "'My name is [your name]'\n\n"
                "Then I can address you personally! 👋"
            )

    match = NAME_PATTERN.search(text)
    if match:
        name = next((g for g in match.groups() if g), None)

        if name and name.lower() not in NOT_A_NAME and len(name) > 2:
            name = name.capitalize()
            _user_name = name
            logger.info(f"User name captured: {name}")

            return (
                f"Namaste {name}! 🌾\n\n"
                f"Nice to meet you! I'm KhedutMitra, your agricultural advisor.\n\n"
                f"I'm here to help with crop management, pest control, irrigation, "
                f"fertilizers, and government schemes.\n\n"
                f"What farming question can I answer for you today, {name}?"
            )

    return None


def clean_query(raw: str) -> str:
    text = raw.strip().lower()
    text = re.sub(r"[^\w\s\?\.\,\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    words = [CROP_CORRECTIONS.get(w, w) for w in words]
    text = " ".join(words)
    for wrong, right in CROP_CORRECTIONS.items():
        text = text.replace(wrong, right)
    logger.info(f"Clean query: {text!r}")
    return text


def retrieve(query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
    store = _load_vectorstore()
    try:
        results = store.similarity_search_with_score(query, k=k)
        converted = []
        for doc, score in results:
            similarity = 1 / (1 + score)
            converted.append((doc, similarity))
        
        all_scores = [round(s, 3) for _, s in converted]
        logger.info(f"Retrieval scores: {all_scores} | Threshold: {MIN_SCORE}")
        
        filtered = [(doc, score) for doc, score in converted if score >= MIN_SCORE]
        logger.info(f"Retrieved {len(filtered)}/{len(converted)} chunks above threshold")
        return filtered
    except Exception as e:
        logger.error(f"Retrieve failed: {e}")
        return []


def build_prompt(
    question: str, 
    docs: List[Tuple[Document, float]], 
    history: List[Tuple[str, str]] = None
) -> str:
    history = history or []
    
    history_text = ""
    if history:
        for role, content in history[-4:]:
            if role == "user":
                history_text += f"User: {content}\n"
            else:
                history_text += f"Assistant: {content}\n"
        history_text += "\n"

    if not docs:
        context = "[No relevant information found in knowledge base.]"
    else:
        context_parts = []
        for i, (doc, score) in enumerate(docs, 1):
            meta = doc.metadata
            crop = meta.get("crop", "general")
            topic = meta.get("topic", "general")
            header = (
                f"[Chunk {i}: {crop.title()} | Topic: {topic.title()} "
                f"| Relevance: {score:.1%}]"
            )
            context_parts.append(f"{header}\n{doc.page_content.strip()}")

        context = "\n\n".join(context_parts)

    final_prompt = PROMPT_TEMPLATE.format(
        conversation_history=history_text,
        context=context,
        question=question,
    )

    logger.info(f"Prompt built ({len(final_prompt)} chars, {len(docs)} chunks)")
    return final_prompt


def _call_ollama(prompt: str) -> Optional[str]:
    try:
        logger.info(f"Calling {OLLAMA_MODEL}…")
        response = chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        answer = response.get("message", {}).get("content", "")
        logger.info(f"LLM response received ({len(answer)} chars)")
        return answer

    except ConnectionError as e:
        logger.error(
            f"Cannot connect to Ollama. Is it running? "
            f"Start with: ollama serve\nError: {e}"
        )
        return None

    except TimeoutError as e:
        logger.error(f"Ollama request timed out ({OLLAMA_TIMEOUT}s): {e}")
        return None

    except Exception as e:
        logger.error(f"Ollama error: {e}", exc_info=True)
        return None


def generate_answer(
    prompt: str,
    docs: List[Tuple[Document, float]],
    history: List[Tuple[str, str]] = None
) -> str:
    llm_response = _call_ollama(prompt)

    if llm_response:
        return llm_response

    if docs:
        logger.warning(
            "LLM unavailable. Returning top retrieved chunk as fallback."
        )
        top_doc, top_score = docs[0]
        content = top_doc.page_content.strip()

        return (
            f"[From our knowledge base - Confidence: {top_score:.1%}]\n\n"
            f"{content}"
        )

    logger.error("LLM failed and no chunks to fall back on.")
    return (
        "I couldn't generate an answer right now. "
        "Please check if Ollama is running (ollama serve) and try again."
    )


def ask(raw_query: str, history: List[Tuple[str, str]] = None) -> dict:
    history = history or []
    timestamp = datetime.now().isoformat()

    logger.info(f"New query: {raw_query!r} | User: {_user_name or 'unknown'}")

    conv_response = _check_conversation(raw_query)
    if conv_response:
        new_history = history + [("user", raw_query), ("assistant", conv_response)]

        return {
            "answer": conv_response,
            "source_chunks": [],
            "clean_query": raw_query.strip().lower(),
            "user_name": _user_name,
            "timestamp": timestamp,
            "is_conversation": True,
        }

    clean = clean_query(raw_query)

    docs = retrieve(clean)
    logger.info(f"Retrieved {len(docs)} relevant chunks")

    prompt = build_prompt(clean, docs, history)

    answer = generate_answer(prompt, docs, history)

    source_chunks = [
        {
            "content": doc.page_content.strip(),
            "crop": doc.metadata.get("crop", "general"),
            "topic": doc.metadata.get("topic", "general"),
            "score": float(round(score, 3)),
        }
        for doc, score in docs
    ]

    new_history = history + [("user", raw_query), ("assistant", answer)]

    return {
        "answer": answer,
        "source_chunks": source_chunks,
        "clean_query": clean,
        "user_name": _user_name,
        "timestamp": timestamp,
        "is_conversation": False,
        "history_length": len(new_history),
    }


if __name__ == "__main__":
    try:
        _load_vectorstore()
        print("✓ Pipeline initialized\n")

        demo_queries = [
            "Hi there!",
            "My name is Raj",
            "How do I grow pomegranates?",
            "What about pest control?",
        ]

        history = []
        for query in demo_queries:
            result = ask(query, history)
            print(f"User: {query}")
            print(f"Answer: {result['answer']}\n")
            history.append(("user", query))
            history.append(("assistant", result["answer"]))

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)