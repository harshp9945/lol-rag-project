"""TF-IDF retriever: the fast, interpretable baseline.

Each chunk becomes a sparse vector of word-importance weights; a query
becomes the same, and cosine similarity ranks by rarity-weighted overlap.
Fast and interpretable, but it matches WORDS not MEANING, which is the
gap the embeddings retriever closes.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import RetrievedChunk, Retriever


class TfidfRetriever(Retriever):
    def __init__(self, ngram_range: tuple[int, int] = (1, 2)):
        self._vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words="english")
        self._matrix = None
        self._chunks: list[dict] = []

    @property
    def name(self) -> str:
        return "tfidf"

    def index(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        self._matrix = self._vectorizer.fit_transform(c["text"] for c in chunks)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        if self._matrix is None:
            raise RuntimeError("index() must be called before retrieve()")
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [
            RetrievedChunk(
                text=self._chunks[i]["text"],
                source=self._chunks[i]["source"],
                score=float(scores[i]),
            )
            for i in top
        ]
