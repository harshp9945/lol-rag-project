"""RAG pipeline: retrieve, guardrail, prompt, generate.

The guardrail is the point of the project: if the best retrieval
similarity is below a per-backend threshold, it refuses instead of
hallucinating. Thresholds differ by backend because similarity scales
differ; recalibrate with evals/calibrate_thresholds.py after any corpus
or chunking change.
"""
from dataclasses import dataclass, field

from .llm.backends import LLMBackend
from .retrieval.base import RetrievedChunk, Retriever

DEFAULT_THRESHOLDS = {
    "tfidf": 0.15,
    "embeddings": 0.35,
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

    def ask(self, question: str) -> RAGResponse:
        chunks = self.retriever.retrieve(question, k=self.k)
        top_score = chunks[0].score if chunks else 0.0
        if top_score < self.threshold:
            return RAGResponse(answer=REFUSAL_MESSAGE, refused=True,
                               chunks=chunks, top_score=top_score)
        context = "\n\n".join(f"[source: {c.source}]\n{c.text}" for c in chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.backend.generate(prompt)
        return RAGResponse(answer=answer, refused=False, chunks=chunks, top_score=top_score)
