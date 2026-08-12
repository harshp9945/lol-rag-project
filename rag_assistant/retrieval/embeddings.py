"""Embeddings retriever: semantic search via sentence-transformers + Chroma.

How it differs from TF-IDF: instead of sparse word-count vectors, each
chunk is mapped by a small neural network (all-MiniLM-L6-v2, 384 dims,
CPU-friendly) into a dense vector where MEANING determines position --
"pick success percentage" lands near "champion win rate" even with zero
shared words. Chroma stores the vectors and does the nearest-neighbour
search.

Trade-offs vs TF-IDF (measure these in evals, don't assume):
  + recall on paraphrased queries
  - slower indexing (model inference per chunk) and query
  - a 90MB model download on first run
  - similarity scores live on a DIFFERENT SCALE, so the refusal
    threshold must be recalibrated per-backend (see pipeline.py).

Requires:  pip install sentence-transformers chromadb
"""
from .base import RetrievedChunk, Retriever

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    _DEPS_OK = True
except ImportError:  # keep the package importable without the heavy deps
    _DEPS_OK = False


class EmbeddingsRetriever(Retriever):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not _DEPS_OK:
            raise ImportError(
                "EmbeddingsRetriever needs extras: "
                "pip install sentence-transformers chromadb"
            )
        self._model = SentenceTransformer(model_name)
        # Ephemeral in-memory collection; swap to PersistentClient for reuse.
        self._collection = chromadb.Client().create_collection(
            name="kb", metadata={"hnsw:space": "cosine"}
        )
        self._indexed = False

    @property
    def name(self) -> str:
        return "embeddings"

    def index(self, chunks: list[dict]) -> None:
        vectors = self._model.encode(
            [c["text"] for c in chunks], show_progress_bar=False
        )
        self._collection.add(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=vectors.tolist(),
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"]} for c in chunks],
        )
        self._indexed = True

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        if not self._indexed:
            raise RuntimeError("index() must be called before retrieve()")
        q_vec = self._model.encode([query]).tolist()
        res = self._collection.query(query_embeddings=q_vec, n_results=k)
        out = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            # Chroma returns cosine DISTANCE (0 = identical); convert to
            # similarity so both backends rank "higher = better".
            out.append(RetrievedChunk(text=text, source=meta["source"], score=1.0 - dist))
        return out
