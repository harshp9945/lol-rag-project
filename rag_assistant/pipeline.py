"""RAG pipeline: retrieve, guardrail, prompt, generate.

The guardrail is the point of the project: if the best retrieval
similarity is below a per-backend threshold, it refuses instead of
hallucinating. Thresholds differ by backend because similarity scales
differ; recalibrate with evals/calibrate_thresholds.py after any corpus
or chunking change.

The retrieve step and the generate step are separate methods so a caller
(for example the API's daily usage cap) can run the free, local retrieval
and guardrail on every request but gate only the LLM generation. ask()
runs both in sequence for the simple case (CLI, tests).
"""
from dataclasses import dataclass, field

from .llm.backends import LLMBackend
from .retrieval.base import RetrievedChunk, Retriever

DEFAULT_THRESHOLDS = {
    "tfidf": 0.15,
    "embeddings": 0.447,
}

REFUSAL_MESSAGE = (
    "I cannot answer that from my knowledge base. The closest material I "
    "have is not similar enough to your question to give a grounded answer."
)

PROMPT_TEMPLATE = """You are an assistant answering questions strictly from the provided context, which contains findings from a League of Legends match-analytics project.

Rules:
- Answer ONLY from the context below. If the context does not contain the answer, say so.
- Cite the source file in brackets after each claim, e.g. [source: champion_balance.md].

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class Retrieval:
    """Result of the retrieval + guardrail step, before any generation."""
    question: str
    chunks: list[RetrievedChunk]
    top_score: float
    refused: bool


@dataclass
class RAGResponse:
    answer: str
    refused: bool
    chunks: list[RetrievedChunk] = field(default_factory=list)
    top_score: float = 0.0


class RAGPipeline:
    def __init__(self, retriever: Retriever, backend: LLMBackend,
                 threshold: float | None = None, k: int = 3):
        self.retriever = retriever
        self.backend = backend
        self.threshold = (
            threshold if threshold is not None
            else DEFAULT_THRESHOLDS.get(retriever.name, 0.2)
        )
        self.k = k

    def retrieve(self, question: str) -> Retrieval:
        """Run retrieval and the guardrail only. Free and local; no LLM call."""
        chunks = self.retriever.retrieve(question, k=self.k)
        top_score = chunks[0].score if chunks else 0.0
        return Retrieval(
            question=question,
            chunks=chunks,
            top_score=top_score,
            refused=top_score < self.threshold,
        )

    def generate(self, retrieval: Retrieval) -> str:
        """Generate an answer for an approved retrieval (makes the LLM call)."""
        context = "\n\n".join(
            f"[source: {c.source}]\n{c.text}" for c in retrieval.chunks
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=retrieval.question)
        return self.backend.generate(prompt)

    def ask(self, question: str) -> RAGResponse:
        """Full pipeline: retrieve, guardrail, then generate if allowed."""
        r = self.retrieve(question)
        if r.refused:
            return RAGResponse(answer=REFUSAL_MESSAGE, refused=True,
                               chunks=r.chunks, top_score=r.top_score)
        answer = self.generate(r)
        return RAGResponse(answer=answer, refused=False,
                           chunks=r.chunks, top_score=r.top_score)
