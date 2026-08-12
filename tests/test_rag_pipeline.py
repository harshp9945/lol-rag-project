"""
test_rag_pipeline.py
---------------------
Evaluation suite for the RAG pipeline. This is the piece most tutorial
RAG projects skip entirely -- and the piece real AI Engineer JDs
specifically call out ("evaluation" was listed as a required skill
alongside RAG and agents in multiple postings).

Three things are tested here, each addressing a different failure mode:

1. RETRIEVAL ACCURACY: does the retriever return the right document
   for a clear, on-topic question?
2. HALLUCINATION GUARDRAIL: does the system correctly refuse to answer
   when the question is off-topic / not in the knowledge base?
3. SOURCE GROUNDING: does every answer that DOES respond include a
   verifiable source citation, so a human can check the claim?

Run with: python -m pytest tests/test_rag_pipeline.py -v
(or run directly: python tests/test_rag_pipeline.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import get_retriever
from generator import answer_question, build_prompt


# ── Test fixtures ──────────────────────────────────────────────────────────────

ON_TOPIC_QUERIES = [
    {
        "query": "What is the win rate for securing the first Baron?",
        "expected_doc_id": "obj_002",
    },
    {
        "query": "Is the game duration normally distributed?",
        "expected_doc_id": "duration_001",
    },
    {
        "query": "What is the AUC of the win prediction model?",
        "expected_doc_id": "model_001",
    },
    {
        "query": "What is Yasuo's ban rate?",
        "expected_doc_id": "balance_002",
    },
]

OFF_TOPIC_QUERIES = [
    "What's the weather like in Summoner's Rift?",
    "Who is the best League of Legends streamer?",
    "What's your favorite pizza topping?",
    "How do I file my taxes in the UK?",
]


# ── Test 1: Retrieval accuracy ──────────────────────────────────────────────────

def test_retrieval_accuracy():
    """For clear on-topic questions, the correct document should be in the top result."""
    retriever = get_retriever("tfidf")
    passed = 0
    failed = []

    for case in ON_TOPIC_QUERIES:
        results = retriever.retrieve(case["query"], top_k=1)
        top_id = results[0]["id"] if results else None
        if top_id == case["expected_doc_id"]:
            passed += 1
        else:
            failed.append({
                "query": case["query"],
                "expected": case["expected_doc_id"],
                "got": top_id,
            })

    accuracy = passed / len(ON_TOPIC_QUERIES)
    print(f"\n=== Retrieval Accuracy: {passed}/{len(ON_TOPIC_QUERIES)} ({accuracy:.0%}) ===")
    for f in failed:
        print(f"  MISS: '{f['query']}' -> expected {f['expected']}, got {f['got']}")

    return accuracy


# ── Test 2: Hallucination guardrail ────────────────────────────────────────────

def test_hallucination_guardrail():
    """For off-topic questions, the system must refuse to answer, not guess."""
    retriever = get_retriever("tfidf")
    passed = 0
    failed = []

    for query in OFF_TOPIC_QUERIES:
        result = answer_question(query, retriever, backend="mock")
        refused = "don't have data" in result["answer"].lower()
        if refused:
            passed += 1
        else:
            failed.append({"query": query, "answer": result["answer"]})

    accuracy = passed / len(OFF_TOPIC_QUERIES)
    print(f"\n=== Hallucination Guardrail: {passed}/{len(OFF_TOPIC_QUERIES)} ({accuracy:.0%}) ===")
    for f in failed:
        print(f"  LEAK: '{f['query']}' -> answered instead of refusing: {f['answer'][:80]}")

    return accuracy


# ── Test 3: Source grounding ────────────────────────────────────────────────────

def test_source_grounding():
    """Every non-refused answer must include at least one verifiable source citation."""
    retriever = get_retriever("tfidf")
    passed = 0
    total = 0

    for case in ON_TOPIC_QUERIES:
        result = answer_question(case["query"], retriever, backend="mock")
        total += 1
        has_source = len(result["sources"]) > 0 and result["sources"][0]["notebook"] is not None
        if has_source:
            passed += 1

    accuracy = passed / total
    print(f"\n=== Source Grounding: {passed}/{total} ({accuracy:.0%}) ===")
    return accuracy


# ── Run all ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running RAG pipeline evaluation suite...")
    retrieval_acc = test_retrieval_accuracy()
    guardrail_acc = test_hallucination_guardrail()
    grounding_acc = test_source_grounding()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Retrieval Accuracy:      {retrieval_acc:.0%}")
    print(f"Hallucination Guardrail: {guardrail_acc:.0%}")
    print(f"Source Grounding:        {grounding_acc:.0%}")

    if retrieval_acc < 0.75 or guardrail_acc < 0.9:
        print("\n⚠ One or more metrics below target threshold -- review retrieval tuning.")
    else:
        print("\n✓ All metrics meet target thresholds.")
