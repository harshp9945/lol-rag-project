"""Tests for the RAG pipeline. Run with: pytest tests/ -v

Uses the mock LLM backend so the suite runs offline and in CI. Each test
targets a specific failure mode: guardrail refusal, retrieval grounding,
chunker edge cases, per-backend threshold defaulting.
"""
import pytest

from rag_assistant.chunker import chunk_text, load_knowledge_base
from rag_assistant.llm.backends import MockBackend
from rag_assistant.pipeline import DEFAULT_THRESHOLDS, RAGPipeline, REFUSAL_MESSAGE
from rag_assistant.retrieval.tfidf import TfidfRetriever

KB_DIR = "knowledge_base"


@pytest.fixture(scope="module")
def indexed_retriever():
    chunks = load_knowledge_base(KB_DIR)
    assert chunks, f"knowledge base at {KB_DIR}/ is empty"
    r = TfidfRetriever()
    r.index(chunks)
    return r


@pytest.fixture()
def pipeline(indexed_retriever):
    return RAGPipeline(retriever=indexed_retriever, backend=MockBackend())


class TestChunker:
    def test_respects_word_ceiling(self):
        long_doc = "\n\n".join(["word " * 80] * 5)
        chunks = chunk_text(long_doc, source="synthetic.md", max_words=120)
        assert all(len(c["text"].split()) <= 160 for c in chunks)
        assert len(chunks) >= 3

    def test_preserves_source_attribution(self):
        assert chunk_text("one para", source="attribution.md")[0]["source"] == "attribution.md"

    def test_empty_document_yields_no_chunks(self):
        assert chunk_text("   \n\n  ", source="empty.md") == []


class TestRetrieval:
    def test_relevant_chunk_ranks_first(self, indexed_retriever):
        hits = indexed_retriever.retrieve("Benjamini-Hochberg correction champion win rates", k=3)
        assert hits[0].source == "champion_balance.md"
        assert hits[0].score >= hits[-1].score

    def test_retrieve_before_index_raises(self):
        with pytest.raises(RuntimeError):
            TfidfRetriever().retrieve("anything")


class TestGuardrail:
    def test_out_of_scope_query_is_refused(self, pipeline):
        resp = pipeline.ask("How do I cook risotto?")
        assert resp.refused is True
        assert resp.answer == REFUSAL_MESSAGE
        assert resp.top_score < pipeline.threshold

    def test_in_scope_query_is_answered_with_context(self, pipeline):
        resp = pipeline.ask("What win rate does taking the first inhibitor give?")
        assert resp.refused is False
        assert "retrieved chunk" in resp.answer
        assert resp.chunks and resp.top_score >= pipeline.threshold

    def test_threshold_defaults_are_backend_specific(self, indexed_retriever):
        p = RAGPipeline(retriever=indexed_retriever, backend=MockBackend())
        assert p.threshold == DEFAULT_THRESHOLDS["tfidf"]

    def test_explicit_threshold_overrides_default(self, indexed_retriever):
        p = RAGPipeline(retriever=indexed_retriever, backend=MockBackend(), threshold=0.99)
        assert p.ask("What win rate does taking the first inhibitor give?").refused is True
