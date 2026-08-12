"""Derive per-backend refusal thresholds empirically.

The guardrail threshold is the score below which the pipeline refuses to
answer. Because similarity scales differ per retriever (TF-IDF cosine
scores on short queries cluster far lower than MiniLM embedding
similarities), a hardcoded threshold is wrong for at least one backend.

Method: run two labelled query sets against the indexed corpus --
  * in_scope:  questions the knowledge base genuinely covers
  * out_of_scope: questions it cannot answer (future events, other games,
    unrelated topics)
-- then report the top-1 score distributions for each and the midpoint
between the lowest in-scope score and the highest out-of-scope score.
If the distributions overlap, the script says so loudly: that means the
corpus or chunking needs work before a clean threshold exists, and the
overlap region is where hallucinations would live.

Usage:
    python -m evals.calibrate_thresholds --retriever tfidf
    python -m evals.calibrate_thresholds --retriever embeddings
"""
import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_assistant.chunker import load_knowledge_base
from rag_assistant.retrieval.tfidf import TfidfRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("calibrate")

QUERY_SETS = Path(__file__).parent / "calibration_queries.json"


def build_retriever(kind: str):
    if kind == "tfidf":
        return TfidfRetriever()
    if kind == "embeddings":
        from rag_assistant.retrieval.embeddings import EmbeddingsRetriever
        return EmbeddingsRetriever()
    raise ValueError(f"unknown retriever '{kind}' (expected tfidf|embeddings)")


def top1_scores(retriever, queries: list[str]) -> list[float]:
    scores = []
    for q in queries:
        hits = retriever.retrieve(q, k=1)
        scores.append(hits[0].score if hits else 0.0)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="tfidf", choices=["tfidf", "embeddings"])
    parser.add_argument("--kb-dir", default="knowledge_base")
    args = parser.parse_args()

    try:
        query_sets = json.loads(QUERY_SETS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.error("calibration_queries.json not found next to this script")
        return 1

    chunks = load_knowledge_base(args.kb_dir)
    if not chunks:
        log.error("no .md/.txt files found in %s -- nothing to index", args.kb_dir)
        return 1
    log.info("indexed %d chunks from %s", len(chunks), args.kb_dir)

    retriever = build_retriever(args.retriever)
    retriever.index(chunks)

    in_scores = top1_scores(retriever, query_sets["in_scope"])
    out_scores = top1_scores(retriever, query_sets["out_of_scope"])

    log.info("[%s] in-scope    top-1: min=%.3f median=%.3f max=%.3f",
             retriever.name, min(in_scores), statistics.median(in_scores), max(in_scores))
    log.info("[%s] out-of-scope top-1: min=%.3f median=%.3f max=%.3f",
             retriever.name, min(out_scores), statistics.median(out_scores), max(out_scores))

    lowest_in, highest_out = min(in_scores), max(out_scores)
    if lowest_in <= highest_out:
        log.warning(
            "DISTRIBUTIONS OVERLAP (lowest in-scope %.3f <= highest out-of-scope %.3f). "
            "No clean threshold exists; improve chunking or corpus coverage first.",
            lowest_in, highest_out,
        )
        return 2

    suggested = round((lowest_in + highest_out) / 2, 3)
    log.info("clean separation -- suggested threshold for '%s': %.3f "
             "(update DEFAULT_THRESHOLDS in rag_assistant/pipeline.py)",
             retriever.name, suggested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
