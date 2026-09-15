"""
Backend API - wraps the retrieval + generation pipeline behind
a FastAPI HTTP API so a frontend (or any client) can send a question
and get back an answer with sources.

Run: uvicorn main:app --reload
     (from inside src/fastAPI/, or use the full module path from elsewhere)
"""
import sys
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT / "src"))

from retrieval.retrieve import load_vectorstore, search 
from generation.generator import generate_answer  

MAX_HISTORY_TURNS = 3

MIN_CONFIDENCE_SCORE = 0.30

NO_CONFIDENT_MATCH_MESSAGE = (
    "اطلاعات کافی و مرتبطی در منابع موجود برای پاسخ به این سؤال پیدا نشد. "
    "لطفاً برای اطلاعات دقیق‌تر با پزشک یا داروساز خود مشورت کنید."
)


app_state: dict = {"conversations": defaultdict(list)}


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
# Allow the frontend (running on a different port, e.g. localhost:3000 or
# a dev server) to call this API from the browser. "*" is fine for local
# development; restrict to specific origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class SourceItem(BaseModel):
    drug_name: str | None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    session_id: str


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the server is up and the index is loaded."""
    return {"status": "ok", "index_loaded": "vectorstore" in app_state}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a Persian pharmaceutical question using the RAG pipeline,
    with short-term memory of the current conversation."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")

    session_id = request.session_id or str(uuid.uuid4())
    history = app_state["conversations"][session_id][-MAX_HISTORY_TURNS:]

    vectorstore = app_state["vectorstore"]
   
    results = search(vectorstore, question)
    best_score = min(score for _, score in results) if results else float("inf")

    if history and history[-1].get("drug_name"):
        prev_drug = history[-1]["drug_name"]
        combined_query = f"در مورد داروی {prev_drug}: {question}"
        combined_results = search(vectorstore, combined_query)
        combined_best = min(score for _, score in combined_results) if combined_results else float("inf")
        if combined_best < best_score:
            results, best_score = combined_results, combined_best

    # Safety net: if even the closest match is too dissimilar, don't
    # let the LLM improvise an answer from weak context - say so directly.
    if best_score > MIN_CONFIDENCE_SCORE:
        answer = NO_CONFIDENT_MATCH_MESSAGE
        results = []
    else:
        answer = generate_answer(question, results, history=history)

    # Save this turn for future requests in the same session.
    app_state["conversations"][session_id].append({"question": question, "answer": answer})

    sources = [
        SourceItem(drug_name=doc.metadata.get("drug_name"), score=float(score))
        for doc, score in results
    ]

    return AskResponse(answer=answer, sources=sources, session_id=session_id)