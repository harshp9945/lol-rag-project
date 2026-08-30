# LoL Analytics RAG Assistant

A retrieval-augmented assistant over the findings of my 51,490-match
League of Legends analytics capstone (Season 9, 2017). It answers only
from the knowledge base and refuses questions the corpus cannot ground.
The refusal guardrail is the point of the project.

Built with AI pair-programming assistance. Every design decision below is
one I can explain and defend.

## Architecture

    knowledge_base/*.md -> chunker -> Retriever (tfidf | embeddings)
                                           |
                        guardrail: top-1 score vs per-backend threshold
                                           |
                           refuse  <-------+-------> prompt -> LLM backend
                                                      (mock | ollama | anthropic)

- One Retriever interface, two implementations. TF-IDF is the fast,
  interpretable baseline; embeddings (MiniLM + Chroma) is the semantic
  upgrade. Both plug into the same contract so evals compare them fairly.
- Per-backend refusal thresholds, derived empirically with
  evals/calibrate_thresholds.py.
- Mock LLM backend so the whole pipeline tests offline with zero API cost.

## What the knowledge base contains

Real findings from the capstone, including: Baron is the most decisive
single objective (81.2% win rate, 89.6% with Tower and Dragon); 46 of 138
champions differ significantly from a 50% win rate after Benjamini-Hochberg
correction; a set of low-ban high-win "stealth OP" champions (Sona, Yorick,
Rammus); and a strong snowball curve where a team behind in towers wins
only ~2.1% of games.

## A real calibration finding

On this corpus, TF-IDF shows a distribution overlap: some in-scope
questions (e.g. "which champions are stealth OP") score below some
out-of-scope gaming questions (e.g. a Valorant or Yasuo question), because
TF-IDF matches words, not meaning, and gaming vocabulary leaks similarity.
No single threshold separates both cleanly. This is the concrete reason the
embeddings retriever is the next step: semantic matching should close the
gap. The calibration script reports this overlap rather than hiding it.

## Quickstart

    pip install -r requirements.txt
    python cli.py                     # offline demo: tfidf + mock backend
    pytest tests/ -v
    python -m evals.calibrate_thresholds --retriever tfidf

    # semantic retrieval + real generation:
    pip install sentence-transformers chromadb
    python cli.py --retriever embeddings --backend ollama

## Status

- [x] Chunker, dual retrievers, guardrail, three LLM backends, CLI, tests
- [x] Real capstone findings across 12 knowledge-base files
- [ ] A few numbers from NB02 and NB06 still marked TODO (first tower/
      dragon/blood rates; model baseline AUCs and feature importances)
- [ ] Embeddings calibration + TF-IDF vs embeddings writeup
- [ ] FastAPI service + Docker + CI (next phase)
