from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_pipeline import ask

app = FastAPI()

# Templates (HTML)
templates = Jinja2Templates(directory="templates")

# Static files (JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    return ask(request.query)