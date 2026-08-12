"""Retriever interface.

Every retrieval backend (TF-IDF, embeddings, anything future) implements
this one contract. The rest of the system -- pipeline, guardrail, evals --
only ever talks to this interface, so retrievers are swappable without
touching any other code. This is the single most important design
decision in the project.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """One chunk of knowledge-base text returned by a retriever."""
    text: str
    source: str          # which knowledge-base file it came from
    score: float         # similarity score (backend-specific scale!)


class Retriever(ABC):
    """Contract: index a corpus once, then answer queries with top-k chunks."""

    @abstractmethod
    def index(self, chunks: list[dict]) -> None:
        """Build the index. Each chunk is {'text': str, 'source': str}."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        """Return the k most relevant chunks, highest score first."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logs and eval reports."""
