"""
ingest.py
---------
Reads all data sources:
  - data/agri_data.txt        (main text file)
  - data/pdfs/*.pdf           (PDF files)
  - data/schemes/*.txt        (government schemes)
  - data/crops/*.txt          (crop-wise files)

Chunks with metadata, generates embeddings,
and builds a FAISS vector index saved to disk.

Run once before starting the server:
    python ingest.py
"""

import os
import re
import json
import pickle
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_DIR    = Path("data")
DATA_PATH   = DATA_DIR / "agri_data.txt"   # main text file
PDF_DIR     = DATA_DIR / "pdfs"            # PDF files folder
SCHEMES_DIR = DATA_DIR / "schemes"         # government schemes folder
CROPS_DIR   = DATA_DIR / "crops"           # crop-wise text files folder

INDEX_PATH    = Path("faiss_index")
CHUNK_SIZE    = 500    # increased from 380 for better context
CHUNK_OVERLAP = 100    # increased from 50 for better continuity

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── Topic & Crop Detection ───────────────────────────────────────────────────

TOPIC_KEYWORDS = {
    "fertilizer":   ["fertilizer", "nitrogen", "phosphorus", "potassium", "urea",
                     "dap", "mop", "nutrient", "zinc", "boron", "sulphur"],
    "pest":         ["pest", "insect", "aphid", "borer", "bollworm", "hopper",
                     "mealybug", "thrips", "whitefly", "armyworm", "spray", "pesticide"],
    "disease":      ["disease", "rust", "blight", "mildew", "smut", "rot",
                     "virus", "fungus", "fungicide", "blast", "wilt"],
    "irrigation":   ["irrigation", "water", "drip", "sprinkler", "moisture",
                     "flood", "tensiometer", "waterlogging"],
    "cultivation":  ["cultivation", "sowing", "variety", "spacing", "yield",
                     "harvest", "transplant", "nursery", "season", "soil"],
    "organic":      ["organic", "vermicompost", "jeevamrit", "neem", "panchagavya",
                     "beejamrit", "compost", "manure"],
    "government":   ["scheme", "pm-kisan", "insurance", "kcc", "msp", "subsidy",
                     "government", "yojana", "enam"],
    "post-harvest": ["storage", "drying", "fumigation", "moisture content",
                     "aflatoxin", "hermetic", "mold"],
    "soil":         ["soil", "ph", "organic matter", "lime", "gypsum", "tillage",
                     "mulching", "crop rotation"],
    "ipm":          ["ipm", "integrated", "biological", "pheromone", "trichogramma",
                     "threshold", "etl", "biocontrol"],
}

CROP_KEYWORDS = [
    "wheat", "rice", "cotton", "maize", "soybean", "corn",
    "basmati", "paddy", "tomato", "potato", "onion", "sugarcane",
    "groundnut", "mustard", "sunflower", "chickpea", "lentil",
    "mango", "banana", "grapes", "pomegranate",
]


def detect_crop(text: str) -> str:
    text_lower = text.lower()
    for crop in CROP_KEYWORDS:
        if crop in text_lower:
            return crop
    return "general"


def detect_topic(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_txt_file(path: Path) -> list[dict]:
    """Load a .txt file and chunk it."""
    print(f"  [txt] Loading: {path}")
    raw_text = path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)

    documents = []
    for i, chunk in enumerate(chunks):
        crop  = detect_crop(chunk)
        topic = detect_topic(chunk)
        documents.append({
            "page_content": chunk.strip(),
            "metadata": {
                "chunk_id": i,
                "crop":     crop,
                "topic":    topic,
                "source":   str(path),
            },
        })
    print(f"  [txt] Created {len(documents)} chunks from {path.name}")
    return documents


def load_pdf_file(path: Path) -> list[dict]:
    """Load a PDF file and chunk it."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        print("  [pdf] PyPDFLoader not available. Run: pip install pypdf")
        return []

    print(f"  [pdf] Loading: {path}")
    try:
        loader = PyPDFLoader(str(path))
        pages  = loader.load()

        # Combine all pages into one text
        full_text = "\n\n".join([p.page_content for p in pages])

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(full_text)

        documents = []
        for i, chunk in enumerate(chunks):
            crop  = detect_crop(chunk)
            topic = detect_topic(chunk)
            documents.append({
                "page_content": chunk.strip(),
                "metadata": {
                    "chunk_id": i,
                    "crop":     crop,
                    "topic":    topic,
                    "source":   str(path),
                },
            })
        print(f"  [pdf] Created {len(documents)} chunks from {path.name}")
        return documents

    except Exception as e:
        print(f"  [pdf] Failed to load {path.name}: {e}")
        return []


def load_all_sources() -> list[dict]:
    """
    Load all data sources:
    - data/agri_data.txt
    - data/pdfs/*.pdf
    - data/schemes/*.txt
    - data/crops/*.txt
    """
    all_documents = []

    # ── 1. Main agri_data.txt ─────────────────────────────────────────────────
    if DATA_PATH.exists():
        print("\n[ingest] Loading main data file...")
        all_documents.extend(load_txt_file(DATA_PATH))
    else:
        print(f"[ingest] WARNING: Main data file not found at '{DATA_PATH}'")

    # ── 2. PDF files ──────────────────────────────────────────────────────────
    if PDF_DIR.exists():
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        if pdf_files:
            print(f"\n[ingest] Loading {len(pdf_files)} PDF file(s) from '{PDF_DIR}'...")
            for pdf_file in pdf_files:
                all_documents.extend(load_pdf_file(pdf_file))
        else:
            print(f"\n[ingest] No PDF files found in '{PDF_DIR}' (folder exists but empty)")
    else:
        print(f"\n[ingest] PDF folder '{PDF_DIR}' not found — skipping")
        print(f"         Create it and add PDFs: mkdir data\\pdfs")

    # ── 3. Government schemes .txt files ──────────────────────────────────────
    if SCHEMES_DIR.exists():
        scheme_files = list(SCHEMES_DIR.glob("*.txt"))
        if scheme_files:
            print(f"\n[ingest] Loading {len(scheme_files)} scheme file(s) from '{SCHEMES_DIR}'...")
            for scheme_file in scheme_files:
                all_documents.extend(load_txt_file(scheme_file))
        else:
            print(f"\n[ingest] No .txt files found in '{SCHEMES_DIR}' (folder exists but empty)")
    else:
        print(f"\n[ingest] Schemes folder '{SCHEMES_DIR}' not found — skipping")
        print(f"         Create it: mkdir data\\schemes")

    # ── 4. Crop-wise .txt files ───────────────────────────────────────────────
    if CROPS_DIR.exists():
        crop_files = list(CROPS_DIR.glob("*.txt"))
        if crop_files:
            print(f"\n[ingest] Loading {len(crop_files)} crop file(s) from '{CROPS_DIR}'...")
            for crop_file in crop_files:
                all_documents.extend(load_txt_file(crop_file))
        else:
            print(f"\n[ingest] No .txt files found in '{CROPS_DIR}' (folder exists but empty)")
    else:
        print(f"\n[ingest] Crops folder '{CROPS_DIR}' not found — skipping")
        print(f"         Create it: mkdir data\\crops")

    return all_documents


# ─── Index Builder ────────────────────────────────────────────────────────────

def build_index(documents: list[dict]) -> None:
    """Embed all chunks and persist FAISS index to disk."""
    print(f"\n[ingest] Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in documents
    ]

    print(f"[ingest] Embedding {len(docs)} chunks…")
    vectorstore = FAISS.from_documents(docs, embeddings)

    INDEX_PATH.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"[ingest] FAISS index saved to '{INDEX_PATH}/'")

    # Save chunk metadata for inspection
    meta_path = INDEX_PATH / "chunks_meta.json"
    meta_path.write_text(
        json.dumps([d["metadata"] for d in documents], indent=2),
        encoding="utf-8",
    )
    print(f"[ingest] Metadata saved to '{meta_path}'")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  KhedutMitra — Data Ingestion Pipeline")
    print("=" * 55)

    # Create folders if they don't exist
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    SCHEMES_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[ingest] Data folders ready:")
    print(f"  ✅ {PDF_DIR}")
    print(f"  ✅ {SCHEMES_DIR}")
    print(f"  ✅ {CROPS_DIR}")

    # Load all sources
    documents = load_all_sources()

    if not documents:
        raise ValueError("No documents found! Add data files and try again.")

    print(f"\n[ingest] Total chunks created: {len(documents)}")

    # Source summary
    from collections import Counter
    sources = Counter(d["metadata"]["source"] for d in documents)
    print("\n[ingest] Chunks per source:")
    for src, count in sources.items():
        print(f"  {count:>4} chunks ← {src}")

    # Sample chunk
    sample = documents[0]
    print("\n── Sample chunk ──────────────────────────────────")
    print(f"  crop  : {sample['metadata']['crop']}")
    print(f"  topic : {sample['metadata']['topic']}")
    print(f"  source: {sample['metadata']['source']}")
    print(f"  text  : {sample['page_content'][:120]}…")
    print("──────────────────────────────────────────────────")

    # Build FAISS index
    build_index(documents)

    print("\n✅ Ingestion complete!")
    print("   Run: uvicorn app:app --reload")


if __name__ == "__main__":
    main()