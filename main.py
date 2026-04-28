from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_pipeline import ask
import re

app = FastAPI()

# Templates (HTML)
templates = Jinja2Templates(directory="templates")

# Static files (JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")


def detect_and_respond(text: str):
    clean = text.strip().lower()

    # Greeting check
    if re.match(r"^(hi|hey|hello|hii|namaste|namaskar|salaam|good morning|good evening)[\s!\.]*$", clean):
        return "Namaste! 🌾 I'm KhedutMitra, your agricultural assistant.\n\nAsk me about:\n• Crop cultivation\n• Fertilizers\n• Pest & disease control\n• Irrigation\n• Government schemes\n\nHow can I help your farm today?"

    # Name introduction check
    match = re.search(r"my\s+name\s+is\s+(\w+)|i\s+am\s+(\w+)|mera\s+naam\s+(\w+)", clean)
    if match:
        name = (match.group(1) or match.group(2) or match.group(3)).capitalize()
        return f"Namaste {name}! 🌾 Welcome to KhedutMitra.\n\nNice to meet you, {name}! Ask me anything about crops, fertilizers, pest control, irrigation, or government farming schemes."

    return None  # not a greeting — pass to RAG


class QueryRequest(BaseModel):
    query: str

# Serve frontend UI
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={}  # add other context variables here if needed
)

# Chat API
@app.post("/chat")
def chat(request: QueryRequest):
    # Check for greetings/name first
    conv_response = detect_and_respond(request.query)
    if conv_response:
        return {"answer": conv_response, "source_chunks": [], "clean_query": request.query}

    # Otherwise go to RAG pipeline
    return ask(request.query)