"""Embeddings retriever: semantic search via a lightweight ONNX model.

Each chunk is mapped by all-MiniLM-L6-v2 into a dense vector where MEANING
determines position, so paraphrases match even with no shared words. This
is what lets the assistant answer "which champions are balance outliers"
when the corpus only says "champions that deviate significantly from 50%".

Implementation notes:
- The model runs through fastembed, which uses ONNX Runtime rather than
  PyTorch. Same MiniLM weights and same output vectors, but a fraction of
  the memory, which is what lets this run on a small (512MB) hosted
  instance. Loading full PyTorch here would exceed that limit.
- For a corpus this small (tens of chunks), a vector database is
  unnecessary overhead. Vectors are held in a numpy array and searched
  with plain cosine similarity, which is exact and effectively instant at
  this scale. If the corpus grew to tens of thousands of chunks, an
  approximate-nearest-neighbour index would be the right move; it is not
  needed here.

Requires:  pip install fastembed numpy
"""
import numpy as np

from .base import RetrievedChunk, Retriever

try:
    from fastembed import TextEmbedding
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingsRetriever(Retriever):
    def __init__(self, model_name: str = _MODEL_NAME):
        if not _DEPS_OK:
            raise ImportError(
                "EmbeddingsRetriever needs extras: pip install fastembed numpy"
            )
        self._model = TextEmbedding(model_name=model_name)
        self._matrix = None            # (n_chunks, dim), L2-normalised
        self._chunks: list[dict] = []

    @property
    def name(self) -> str:
        return "embeddings"

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings so a dot product equals cosine similarity."""
        vecs = np.array(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # guard against divide-by-zero
        return vecs / norms

    def index(self, chunks: list[dict]) -> None:
        self._chunks = chunks
        self._matrix = self._embed([c["text"] for c in chunks])

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        if self._matrix is None:
            raise RuntimeError("index() must be called before retrieve()")
        q = self._embed([query])[0]                 # (dim,), normalised
        scores = self._matrix @ q                    # cosine similarity per chunk
        top = np.argsort(scores)[::-1][:k]
        return [
            RetrievedChunk(
                text=self._chunks[i]["text"],
                source=self._chunks[i]["source"],
                score=float(scores[i]),
            )
            for i in top
        ]
