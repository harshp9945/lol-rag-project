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
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_assistant.chunker import load_knowledge_base
from rag_assistant.llm.backends import MockBackend, OllamaBackend
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.retrieval.tfidf import TfidfRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
log = logging.getLogger("api")

KB_DIR = "knowledge_base"

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
    """Prefer local Ollama (free, no API key), fall back to mock so the
    service always starts even if Ollama is not installed or running.

    Ollama gives real generated answers with no paid API key, which makes
    the project runnable for anyone willing to install a free local tool.
    If Ollama is unreachable, the mock backend keeps retrieval and the
    guardrail fully testable offline.
    """
    try:
        backend = OllamaBackend()
        # Probe that the Ollama server is actually reachable before committing
        # to it, so we fall back cleanly rather than failing on the first query.
        backend.generate("ping")
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


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a question from the knowledge base, or refuse if ungrounded."""
    resp = _pipeline.ask(req.question)
    return AskResponse(
        question=req.question,
        answer=resp.answer,
        refused=resp.refused,
        top_score=round(resp.top_score, 4),
        sources=sorted({c.source for c in resp.chunks}),
    )
