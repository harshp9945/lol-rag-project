"""
retriever.py
------------
The retrieval half of RAG: given a user question, find the most relevant
knowledge documents to ground the LLM's answer.

DESIGN NOTE FOR THE PORTFOLIO / INTERVIEW STORY:
This file intentionally supports TWO retrieval backends:

1. TF-IDF (scikit-learn) -- works completely offline, free, no API key
   needed. Good enough to demo the RAG *pattern* correctly.
2. Embedding-based (sentence-transformers or OpenAI/Anthropic embeddings)
   -- the production-grade approach, captures semantic similarity rather
   than just keyword overlap.

Building both, and being able to explain the tradeoff (speed/cost vs.
semantic quality), is a much stronger interview answer than only ever
having used one black-box library. This is the kind of decision a real
AI Engineer JD evaluation panel will probe.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"


def load_documents() -> List[Dict]:
    """Load all knowledge base documents (excluding the manifest)."""
    docs = []
    for filepath in sorted(DOCS_DIR.glob("*.json")):
        if filepath.name == "_manifest.json":
            continue
        with open(filepath) as f:
            docs.append(json.load(f))
    return docs


class TFIDFRetriever:
    """
    Lightweight, fully local retriever. No API key, no network call.
    Good for development and for demonstrating the RAG pattern without
    incurring API costs on every test run.

    Limitation to be upfront about: TF-IDF matches on word overlap, not
    meaning. "Win rate for securing Baron" and "How often does the Baron
    team win" will match less well than with real embeddings, since the
    word overlap is lower despite near-identical meaning.
    """

    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [doc["content"] for doc in documents]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Return the top_k most relevant documents for a query, with similarity scores."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()

        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            doc = self.documents[idx].copy()
            doc["similarity_score"] = round(float(similarities[idx]), 4)
            results.append(doc)
        return results


class EmbeddingRetriever:
    """
    Production-grade retriever using real semantic embeddings.

    Requires: pip install sentence-transformers
    (Not run in this sandboxed environment due to no network access,
    but this is the version to actually deploy.)

    Uses a small, fast, free local embedding model (all-MiniLM-L6-v2)
    rather than calling an API for every query -- cheaper and faster
    for a knowledge base this size (14 documents).
    """

    def __init__(self, documents: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # local import: optional dependency

        self.documents = documents
        self.model = SentenceTransformer(model_name)
        corpus = [doc["content"] for doc in documents]
        self.doc_embeddings = self.model.encode(corpus, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, self.doc_embeddings).flatten()

        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            doc = self.documents[idx].copy()
            doc["similarity_score"] = round(float(similarities[idx]), 4)
            results.append(doc)
        return results


def get_retriever(backend: str = "tfidf"):
    """Factory function -- swap backends with one line."""
    documents = load_documents()
    if backend == "tfidf":
        return TFIDFRetriever(documents)
    elif backend == "embedding":
        return EmbeddingRetriever(documents)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'tfidf' or 'embedding'.")


if __name__ == "__main__":
    retriever = get_retriever("tfidf")
    test_queries = [
        "What's the win rate for securing Baron?",
        "Is Yasuo overpowered?",
        "How accurate is the win prediction model?",
        "Can teams come back from being behind?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, top_k=2)
        for r in results:
            print(f"  [{r['similarity_score']}] {r['id']}: {r['content'][:100]}...")
