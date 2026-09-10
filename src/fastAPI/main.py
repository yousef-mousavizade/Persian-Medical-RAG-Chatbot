"""
Backend API - wraps the retrieval + generation pipeline behind
a FastAPI HTTP API so a frontend (or any client) can send a question
and get back an answer with sources.

Run: uvicorn main:app --reload
    (from inside src/api/, or use the full module path from elsewhere)

Requires: pip install fastapi uvicorn
          (plus everything retrieve.py and generator.py already need)
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Add "src" itself (not the individual subfolders) so package-style
# imports like "retrieval.retrieve" work below. This requires an
# __init__.py file inside each of src/retrieval/ and src/generation/
# (even an empty one) so Python recognizes them as packages.
sys.path.append(str(PROJECT_ROOT / "src"))

from retrieval.retrieve import load_vectorstore, search  
from generation.generator import generate_answer  



app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the FAISS index + embedding model exactly once.
    print("Loading FAISS index and embedding model...")
    app_state["vectorstore"] = load_vectorstore()
    print("Ready to serve requests.")
    yield
    
    app_state.clear()


app = FastAPI(
    title="Persian Medical RAG Chatbot API",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    drug_name: str | None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the server is up and the index is loaded."""
    return {"status": "ok", "index_loaded": "vectorstore" in app_state}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a Persian pharmaceutical question using the RAG pipeline."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    vectorstore = app_state["vectorstore"]
    results = search(vectorstore, question)
    answer = generate_answer(question, results)

    sources = [
        SourceItem(drug_name=doc.metadata.get("drug_name"), score=float(score))
        for doc, score in results
    ]

    return AskResponse(answer=answer, sources=sources)