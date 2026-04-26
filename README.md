# 🌾 Agri RAG Chatbot

A production-quality Agricultural AI Chatbot built with a **Retrieval-Augmented Generation (RAG)** pipeline. Every answer is grounded in retrieved knowledge — no hallucinations.

---

## 📁 Project Structure

```
agri-rag-chatbot/
├── app.py              # FastAPI backend — routes, request/response models
├── rag_pipeline.py     # Full RAG pipeline (clean → embed → retrieve → generate)
├── ingest.py           # One-time data ingestion — chunk, embed, build FAISS index
├── data/
│   └── agri_data.txt   # Curated agricultural knowledge base
├── templates/
│   └── index.html      # Chat UI (earthy design, example chips, source accordion)
├── static/
│   └── script.js       # Chat logic, voice input (Web Speech API), source display
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API key (optional but recommended)
```bash
cp .env.example .env
# Edit .env and add your OpenAI key
# Without it, the chatbot uses retrieval-only fallback (still works!)
```

### 3. Ingest data (run once)
```bash
python ingest.py
```
This reads `data/agri_data.txt`, chunks it, generates embeddings, and saves a FAISS index to `faiss_index/`.

### 4. Start the server
```bash
uvicorn app:app --reload --port 8000
```

### 5. Open in browser
```
http://localhost:8000
```

---

## 🧠 What is RAG?

**RAG (Retrieval-Augmented Generation)** is an AI architecture that grounds LLM responses in real retrieved documents rather than relying solely on the model's training data.

```
User Query
    │
    ▼
Query Cleaning          ← lowercase, strip noise
    │
    ▼
Embedding Generation    ← convert query to vector (MiniLM-L6-v2)
    │
    ▼
Vector Search (FAISS)   ← find top-3 most similar knowledge chunks
    │
    ▼
Prompt Assembly         ← inject chunks into strict prompt template
    │
    ▼
LLM Generation          ← GPT-3.5 generates answer ONLY from context
    │
    ▼
Return Answer + Sources
```

---

## 🔁 Why RAG Instead of Fine-Tuning?

| Dimension          | RAG                          | Fine-Tuning                    |
|--------------------|------------------------------|--------------------------------|
| **Data updates**   | Just re-run `ingest.py`      | Re-train the model             |
| **Cost**           | Near zero                    | Expensive GPU compute          |
| **Hallucination**  | Controlled (context-bound)   | Still possible                 |
| **Explainability** | Show exact source chunks     | Black box                      |
| **Speed to deploy**| Hours                        | Days/weeks                     |
| **Best for**       | Domain Q&A, knowledge bases  | Style/tone adaptation          |

For an agricultural knowledge assistant with a curated dataset that needs frequent updates, **RAG is the right choice**.

---

## 🔍 How Hallucination is Reduced

1. **Strict prompt**: LLM is instructed to answer **ONLY using provided context**.
2. **Similarity threshold**: Chunks below `score < 0.30` are discarded.
3. **Explicit fallback**: If no chunks meet the threshold → returns `"Insufficient data"`.
4. **Low temperature**: LLM called at `temperature=0.2` for factual, conservative answers.
5. **Source transparency**: Every answer shows the exact retrieved chunks used.

---

## 🔌 API

### `POST /ask`

**Request:**
```json
{ "query": "What fertilizer should I apply to wheat?" }
```

**Response:**
```json
{
  "answer": "For wheat, apply...",
  "source_chunks": [
    {
      "content": "For wheat, apply basal dose of 60 kg Nitrogen...",
      "crop": "wheat",
      "topic": "fertilizer",
      "score": 0.82
    }
  ],
  "clean_query": "what fertilizer should i apply to wheat?",
  "latency_ms": 340.5
}
```

### `GET /health`
```json
{ "status": "ok", "service": "agri-rag-chatbot" }
```

---

## 🌾 Chunking Strategy

| Parameter     | Value         | Reason                                         |
|---------------|---------------|------------------------------------------------|
| `chunk_size`  | 380 chars     | ~95 tokens; fits one topic without overflow    |
| `chunk_overlap`| 50 chars     | Prevents context loss at chunk boundaries      |
| `separators`  | `\n\n, \n, .` | Respects paragraph/sentence boundaries first  |
| Metadata      | crop + topic  | Enables future filtered retrieval              |

---

## 🗣️ Voice Input

Uses the **Web Speech API** (built into modern browsers — no library needed).

- Click the 🎙️ button to start listening
- Speak your question in English/Hindi-English
- Auto-sends when speech is detected as final
- Language: `en-IN` (Indian English)

**Supported browsers:** Chrome, Edge, Safari 14.1+  
**Not supported:** Firefox (no Web Speech API)

---

## ⚖️ Trade-offs

| Trade-off | Decision Made |
|-----------|---------------|
| Local FAISS vs cloud vector DB | FAISS — simple, no infra, perfect for resume/demo |
| OpenAI vs local LLM | OpenAI (with free fallback) — best quality |
| sentence-transformers vs OpenAI embeddings | sentence-transformers — free, fast, good quality |
| React vs plain HTML/JS | Plain HTML/JS — simpler stack, easier to explain |
| Complex metadata filtering | Not implemented — KISS principle |

---

## 💡 Example Queries

```
What fertilizer should I use for wheat crop?
How to control pink bollworm in cotton?
What is the best irrigation method for rice?
How to manage blast disease in rice?
What government schemes are available for farmers?
How to improve soil health organically?
What is the water requirement for maize?
How to store wheat grain safely?
What is Integrated Pest Management?
How do I treat yellow rust in wheat?
```

---

## 🏗️ Extending the System

**Add more data:** Append to `data/agri_data.txt` and re-run `python ingest.py`.

**Add more crops:** Add sections to `agri_data.txt` following the `== CROP NAME ==` pattern.

**Switch to local LLM:** Replace `_call_openai()` in `rag_pipeline.py` with Ollama or LlamaCpp.

**Add PDF ingestion:** Use `langchain_community.document_loaders.PyPDFLoader` in `ingest.py`.

---

## 📖 Interview Explanation (30-second version)

> "This is a RAG system. Instead of fine-tuning an LLM on agricultural data — which is expensive and inflexible — I pre-process a curated knowledge base into overlapping text chunks, embed them with a sentence transformer, and store them in FAISS. When a farmer asks a question, I embed their query, find the 3 most similar knowledge chunks via cosine similarity, and inject only those chunks into a strict prompt that forbids the LLM from using outside knowledge. This means the answer is always traceable to a source, and if the data isn't there, we say 'Insufficient data' rather than hallucinating."

---

## 🛠️ Tech Stack

| Layer       | Technology                           |
|-------------|--------------------------------------|
| Backend     | Python 3.10+, FastAPI                |
| RAG         | LangChain                            |
| Embeddings  | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB   | FAISS (local)                        |
| LLM         | OpenAI GPT-3.5-turbo                 |
| Frontend    | HTML5, CSS3, Vanilla JS              |
| Voice       | Web Speech API                       |
