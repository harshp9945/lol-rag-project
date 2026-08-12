"""TF-IDF retriever: the fast, zero-dependency-beyond-sklearn baseline.

How it works, in one breath: every chunk becomes a sparse vector of
word-importance weights (words frequent in THIS chunk but rare across
the corpus score high). A query becomes the same kind of vector, and
cosine similarity ranks chunks by word overlap weighted by rarity.

Strengths: instant indexing, millisecond queries, fully interpretable.
Weakness: it matches WORDS, not MEANING -- "champion win rate" will not
match a chunk that only says "pick success percentage". That semantic
gap is exactly what the embeddings retriever exists to close.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import RetrievedChunk, Retriever


class TfidfRetriever(Retriever):
    def __init__(self, ngram_range: tuple[int, int] = (1, 2)):
        # Unigrams + bigrams: bigrams let "win rate" behave as one term,
        # which noticeably improves precision on stat-heavy text.
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
