"""FastAPI service exposing the RAG assistant over HTTP.

This wraps the exact same RAGPipeline the CLI uses, behind a web endpoint,
so any client can ask questions as JSON and get grounded, guardrailed
answers back. The pipeline logic is unchanged; this just puts a web door
on the front of it.

Design choices worth knowing:
- The pipeline is built ONCE at startup, not per request. Loading the
  embeddings model on every call would be slow and wasteful; here the
  model is loaded once and reused for the life of the server.
- Defaults to the embeddings retriever (its calibration showed clean
  separation) and to a local Ollama backend (free, no API key), falling
  back to the offline mock so the service never fails to start.
- Pydantic models validate every incoming request, so a malformed body
  returns a clean 422 error instead of crashing the handler.

Run locally:
    pip install fastapi uvicorn
    uvicorn api:app --reload
    # then open http://127.0.0.1:8000/docs for the interactive UI
"""
import logging
import os
from datetime import date
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_assistant.chunker import load_knowledge_base
from rag_assistant.llm.backends import GeminiBackend, MockBackend, OllamaBackend
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.retrieval.tfidf import TfidfRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
log = logging.getLogger("api")

KB_DIR = "knowledge_base"

# ---- daily usage cap -------------------------------------------------------
# Caps how many questions the hosted demo will generate real answers for per
# day, so it can never exceed the free API tier or run up a cost. When the
# cap is hit, the endpoint still runs retrieval and the guardrail but returns
# a friendly "limit reached" message instead of calling the LLM. Set via the
# DAILY_QUESTION_CAP env var; defaults to 100.
DAILY_QUESTION_CAP = int(os.getenv("DAILY_QUESTION_CAP", "100"))
_usage = {"day": date.today(), "count": 0}


def _within_daily_cap() -> bool:
    """Return True if a generated answer is still allowed today; count it if so."""
    today = date.today()
    if _usage["day"] != today:
        _usage["day"] = today
        _usage["count"] = 0
    if _usage["count"] >= DAILY_QUESTION_CAP:
        return False
    _usage["count"] += 1
    return True

# ---- request / response schemas -------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask the assistant.")


class SourceChunk(BaseModel):
    source: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    top_score: float
    sources: list[str]


# ---- app + startup ---------------------------------------------------------

app = FastAPI(
    title="LoL Analytics RAG Assistant",
    description="Ask questions about a 51K-match League of Legends analytics "
                "capstone. Answers only from the knowledge base; refuses "
                "questions it cannot ground.",
    version="1.0.0",
)

# Module-level holder for the single shared pipeline.
_pipeline: RAGPipeline | None = None


def _select_backend():
    """Pick the best available backend for real generated answers.

    Order of preference:
      1. Gemini (free-tier hosted API) if GEMINI_API_KEY is set. This is
         what the public deployment uses, so visitors get real answers.
      2. Local Ollama if it is running (free, for local development).
      3. Mock backend, so the service always starts even with neither.
    """
    import os
    if os.getenv("GEMINI_API_KEY"):
        try:
            backend = GeminiBackend()
            return backend
        except Exception as exc:
            log.warning("Gemini unavailable (%s); trying Ollama", exc)
    try:
        backend = OllamaBackend()
        backend.generate("ping")  # probe reachability before committing
        return backend
    except Exception as exc:
        log.warning("Ollama unavailable (%s); using mock backend", exc)
        return MockBackend()


def _build_pipeline() -> RAGPipeline:
    """Build the default pipeline once: embeddings retriever + best backend."""
    try:
        from rag_assistant.retrieval.embeddings import EmbeddingsRetriever
        retriever = EmbeddingsRetriever()
    except Exception as exc:
        log.warning("Embeddings retriever unavailable (%s); using TF-IDF", exc)
        retriever = TfidfRetriever()

    chunks = load_knowledge_base(KB_DIR)
    if not chunks:
        raise RuntimeError(f"knowledge base '{KB_DIR}' is empty")
    retriever.index(chunks)

    backend = _select_backend()
    pipe = RAGPipeline(retriever, backend)
    log.info("pipeline ready: retriever=%s backend=%s chunks=%d threshold=%.3f",
             retriever.name, backend.name, len(chunks), pipe.threshold)
    return pipe


@app.on_event("startup")
def _startup() -> None:
    global _pipeline
    _pipeline = _build_pipeline()


# ---- endpoints -------------------------------------------------------------

@app.get("/")
def home() -> FileResponse:
    """Serve the friendly web UI."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    """Liveness check. Returns which retriever and backend are active."""
    if _pipeline is None:
        return {"status": "starting"}
    return {
        "status": "ok",
        "retriever": _pipeline.retriever.name,
        "backend": _pipeline.backend.name,
        "threshold": _pipeline.threshold,
    }


CAP_MESSAGE = (
    "The live demo has reached today's free question limit (it runs on a free "
    "API tier with a daily cap to keep it truly free). The retrieval and "
    "grounding guardrail below are still real; the written answer resets "
    "tomorrow. To use it without limits, run the project locally, see the repo."
)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question from the knowledge base, or refuse if ungrounded.

    Retrieval and the guardrail always run (they are local and free). Only
    the LLM generation step is subject to the daily cap, so even when the cap
    is hit the caller still sees real sources, scores, and refusals.
    """
    retrieval = _pipeline.retrieve(req.question)

    if retrieval.refused:
        answer, refused = _pipeline_refusal_message(), True
    elif _within_daily_cap():
        answer, refused = _pipeline.generate(retrieval), False
    else:
        answer, refused = CAP_MESSAGE, False

    return AskResponse(
        question=req.question,
        answer=answer,
        refused=refused,
        top_score=round(retrieval.top_score, 4),
        sources=sorted({c.source for c in retrieval.chunks}),
    )


def _pipeline_refusal_message() -> str:
    from rag_assistant.pipeline import REFUSAL_MESSAGE
    return REFUSAL_MESSAGE
