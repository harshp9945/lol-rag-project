"""Tests for the FastAPI layer.

Run with:  pytest tests/test_api.py -v

These use FastAPI's TestClient, which exercises the real app (routing,
Pydantic validation, response shaping) in-process without starting a
server. To keep the suite fast, offline, and free, the pipeline is
overridden with a TF-IDF + mock backend so no model download or Ollama
server is needed.
"""
import pytest
from fastapi.testclient import TestClient

import api
from rag_assistant.chunker import load_knowledge_base
from rag_assistant.llm.backends import MockBackend
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.retrieval.tfidf import TfidfRetriever


@pytest.fixture(scope="module", autouse=True)
def test_pipeline():
    """Replace the module pipeline with a fast offline one for tests."""
    chunks = load_knowledge_base(api.KB_DIR)
    r = TfidfRetriever()
    r.index(chunks)
    api._pipeline = RAGPipeline(r, MockBackend())
    yield
    api._pipeline = None


@pytest.fixture()
def client():
    return TestClient(api.app)


def test_health_reports_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "retriever" in body and "backend" in body


def test_ask_returns_structured_answer(client):
    resp = client.post("/ask", json={"question": "What win rate does first Baron give?"})
    assert resp.status_code == 200
    body = resp.json()
    # Response schema is enforced by Pydantic; check the fields exist and cohere.
    assert body["question"]
    assert "answer" in body
    assert isinstance(body["refused"], bool)
    assert isinstance(body["sources"], list)


def test_empty_question_is_rejected(client):
    # min_length=1 on the request model should trigger a 422 validation error.
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_missing_question_field_is_rejected(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 422
