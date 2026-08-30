"""Derive per-backend refusal thresholds empirically.

Runs two labelled query sets (in_scope, out_of_scope) against the indexed
corpus and reports top-1 score distributions. Suggests a threshold at the
midpoint if the distributions separate cleanly; warns loudly if they
overlap, which means the corpus or chunking needs work.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("calibrate")

QUERY_SETS = Path(__file__).parent / "calibration_queries.json"


def build_retriever(kind: str):
    if kind == "tfidf":
        return TfidfRetriever()
    if kind == "embeddings":
        from rag_assistant.retrieval.embeddings import EmbeddingsRetriever
        return EmbeddingsRetriever()
    raise ValueError(f"unknown retriever '{kind}'")


def top1_scores(retriever, queries):
    return [ (retriever.retrieve(q, k=1)[0].score if retriever.retrieve(q, k=1) else 0.0) for q in queries ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", default="tfidf", choices=["tfidf", "embeddings"])
    parser.add_argument("--kb-dir", default="knowledge_base")
    args = parser.parse_args()

    try:
        query_sets = json.loads(QUERY_SETS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.error("calibration_queries.json not found")
        return 1

    chunks = load_knowledge_base(args.kb_dir)
    if not chunks:
        log.error("no files in %s", args.kb_dir)
        return 1
    log.info("indexed %d chunks from %s", len(chunks), args.kb_dir)

    retriever = build_retriever(args.retriever)
    retriever.index(chunks)

    in_scores = top1_scores(retriever, query_sets["in_scope"])
    out_scores = top1_scores(retriever, query_sets["out_of_scope"])

    log.info("[%s] in-scope     top-1: min=%.3f median=%.3f max=%.3f",
             retriever.name, min(in_scores), statistics.median(in_scores), max(in_scores))
    log.info("[%s] out-of-scope top-1: min=%.3f median=%.3f max=%.3f",
             retriever.name, min(out_scores), statistics.median(out_scores), max(out_scores))

    lowest_in, highest_out = min(in_scores), max(out_scores)
    if lowest_in <= highest_out:
        log.warning("DISTRIBUTIONS OVERLAP (lowest in-scope %.3f <= highest out-of-scope %.3f). "
                    "Consider a stricter threshold or better corpus coverage.", lowest_in, highest_out)
        return 2

    suggested = round((lowest_in + highest_out) / 2, 3)
    log.info("clean separation. suggested threshold for '%s': %.3f", retriever.name, suggested)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
