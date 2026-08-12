"""The RAG pipeline: retrieve -> guardrail -> prompt -> generate.

The guardrail is the part worth explaining in interviews. A naive RAG
system answers EVERY question: ask it "who wins Worlds 2027?" and it
happily hallucinates around whatever weakly-related chunks it fetched.
The guardrail refuses instead: if the best retrieval similarity is
below a threshold, the honest answer is "my knowledge base doesn't
cover that."

CRITICAL calibration note: similarity scores are backend-specific.
TF-IDF cosine similarities on short queries cluster low (a good match
might score 0.2); MiniLM embedding similarities cluster high (an
unrelated pair can still score 0.4). A threshold tuned for one backend
is meaningless for the other -- which is why thresholds live in a
per-backend table, derived empirically by running the eval suite's
out-of-scope queries and finding the score gap between answerable and
unanswerable. Re-run scripts in evals/ to re-derive after any corpus
or chunking change.
"""
from dataclasses import dataclass, field

from .llm.backends import LLMBackend
from .retrieval.base import RetrievedChunk, Retriever

# Empirically derived starting points -- re-calibrate with evals/threshold_calibration.py
DEFAULT_THRESHOLDS = {
    "tfidf": 0.15,
    "embeddings": 0.35,
}

REFUSAL_MESSAGE = (
    "I can't answer that from my knowledge base -- the closest material "
    "I have isn't similar enough to your question to give a grounded answer."
)

PROMPT_TEMPLATE = """You are an assistant answering questions strictly from the provided context, which contains findings from a League of Legends match-analytics project.

Rules:
- Answer ONLY from the context below. If the context does not contain the answer, say so.
- Cite the source file in brackets after each claim, e.g. [source: champion_winrates.md].

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
    def __init__(
        self,
        retriever: Retriever,
        backend: LLMBackend,
        threshold: float | None = None,
        k: int = 3,
    ):
        self.retriever = retriever
        self.backend = backend
        # Per-backend threshold unless explicitly overridden.
        self.threshold = (
            threshold
            if threshold is not None
            else DEFAULT_THRESHOLDS.get(retriever.name, 0.2)
        )
        self.k = k

    def ask(self, question: str) -> RAGResponse:
        chunks = self.retriever.retrieve(question, k=self.k)
        top_score = chunks[0].score if chunks else 0.0

        # ---- the guardrail ----
        if top_score < self.threshold:
            return RAGResponse(
                answer=REFUSAL_MESSAGE, refused=True, chunks=chunks, top_score=top_score
            )

        context = "\n\n".join(
            f"[source: {c.source}]\n{c.text}" for c in chunks
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self.backend.generate(prompt)
        return RAGResponse(answer=answer, refused=False, chunks=chunks, top_score=top_score)
