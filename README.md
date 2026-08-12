# LoL Analytics RAG Assistant

Retrieval-augmented assistant over the findings of my 51K-match League
of Legends analytics capstone. Ask it what the analysis found; it
answers **only** from the knowledge base and **refuses** questions the
corpus can't ground — the refusal guardrail is the point of the project.

Built with AI pair-programming assistance; every design decision below
is one I can defend line by line.

## Architecture

    knowledge_base/*.md --> chunker --> Retriever (tfidf | embeddings)
                                             |
                          guardrail: top-1 score vs per-backend threshold
                                             |
                             refuse  <-------+-------> prompt --> LLM backend
                                                        (mock | ollama | anthropic)

Design decisions worth knowing about:

- **One `Retriever` interface, two implementations.** TF-IDF (sklearn)
  is the fast, interpretable baseline; embeddings (MiniLM + Chroma)
  closes the paraphrase gap. Evals run against both through the same
  contract.
- **Per-backend refusal thresholds.** Similarity scales differ by
  backend, so thresholds are derived empirically per retriever with
  `evals/calibrate_thresholds.py`, which refuses to suggest a number
  when the in-scope / out-of-scope score distributions overlap.
- **Mock LLM backend.** The full pipeline (chunking, retrieval,
  guardrail, prompt assembly) tests offline and in CI with zero API
  calls; the mock's output encodes how many retrieved chunks reached
  the prompt, so grounding is assertable.

## Quickstart

    pip install -r requirements.txt
    python cli.py                     # offline demo: tfidf + mock backend
    pytest tests/ -v                  # test suite

    # semantic retrieval + real generation:
    pip install sentence-transformers chromadb
    python cli.py --retriever embeddings --backend ollama

## Calibrating the guardrail

    python -m evals.calibrate_thresholds --retriever tfidf

Run this after ANY corpus or chunking change. Example of why it exists:
on a 3-document toy corpus, "Which agent should I main in Valorant?"
(out-of-scope) scored 0.163 while a legitimate in-scope query scored
0.150 — shared gaming vocabulary leaks similarity, and a fixed threshold
would either hallucinate or over-refuse. The script surfaces that
overlap instead of hiding it.

## Status

- [x] Chunker, dual retrievers, guardrail, 3 LLM backends, CLI, tests
- [ ] Replace placeholder knowledge base with real capstone findings
- [ ] Recalibrate thresholds on the real corpus (both backends)
- [ ] TF-IDF vs embeddings comparison writeup (`docs/retriever_comparison.md`)
- [ ] FastAPI service + Docker + CI (next phase of the roadmap)
