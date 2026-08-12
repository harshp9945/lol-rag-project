"""
generator.py
------------
The "generation" half of RAG: takes retrieved documents + the user's
question, and produces a grounded natural-language answer.

THE CORE DESIGN PROBLEM THIS FILE SOLVES:
LLMs already "know" things about League of Legends from training data,
which may be outdated, generic, or simply wrong for THIS specific
dataset (Season 9, 51,490 matches). The entire point of RAG is to force
the model to answer ONLY from the retrieved documents, not its own
internal knowledge. This file is where that constraint is actually
enforced -- through prompt design and an explicit "I don't know"
fallback when retrieval comes back empty or low-confidence.

This is the single most important engineering decision in a RAG system,
and the one most tutorials skip. Be ready to explain this exact tradeoff
in an interview: how do you stop the model from confidently making
things up when the answer isn't actually in your knowledge base?
"""

import os
from typing import List, Dict


SYSTEM_PROMPT = """You are a data analyst assistant for a League of Legends match analytics project.

You must answer questions ONLY using the information provided in the CONTEXT below.
Do not use any outside knowledge about League of Legends, even if you know it from training.

Rules:
1. If the context fully answers the question, answer clearly and cite the source notebook.
2. If the context partially answers the question, answer what you can and explicitly state what's missing.
3. If the context does NOT contain relevant information, say "I don't have data on that in this analysis" -- do NOT guess or use outside knowledge.
4. Always cite the source_notebook field when you state a specific number or finding.
5. Keep answers concise -- 2-4 sentences unless the question asks for detail.
"""


def build_prompt(query: str, retrieved_docs: List[Dict], min_similarity: float = 0.15) -> str:
    """
    Build the full prompt sent to the LLM, including retrieved context.

    The min_similarity threshold is a deliberate guardrail: if NOTHING
    retrieved is even weakly relevant, we tell the model explicitly that
    context is empty, rather than passing in noise that invites the
    model to hallucinate a connection that isn't there.

    THRESHOLD TUNING NOTE (real finding from building this project):
    With TF-IDF, a threshold of 0.05 was too permissive -- a deliberately
    irrelevant test query ("What's the weather like in Summoner's Rift?")
    still scored 0.1229 against an unrelated document purely due to common
    word overlap ("like"), and would have produced a confidently wrong
    answer instead of "I don't know." Raising the threshold to 0.15
    filters this out. This is a concrete example of why retrieval
    threshold tuning matters in production RAG systems, not just having
    a threshold at all. With embedding-based retrieval this exact failure
    mode is rarer since semantic similarity doesn't spike on stopword
    overlap, but the same tuning discipline still applies.
    """
    relevant_docs = [d for d in retrieved_docs if d.get("similarity_score", 1.0) >= min_similarity]

    if not relevant_docs:
        context_block = "[No relevant documents found in the knowledge base for this query.]"
    else:
        context_block = "\n\n".join(
            f"[Source: {d['source_notebook']} | Topic: {d['topic']}]\n{d['content']}"
            for d in relevant_docs
        )

    prompt = f"""CONTEXT:
{context_block}

QUESTION:
{query}

Answer based only on the CONTEXT above."""
    return prompt


def generate_answer_ollama(query: str, retrieved_docs: List[Dict], model: str = "llama3") -> str:
    """
    Generate a grounded answer using Ollama -- runs a real LLM completely
    locally and for free. No API key, no per-call cost, no internet
    needed once the model is downloaded.

    Setup (one-time):
        1. Install Ollama: https://ollama.com/download
        2. Pull a model:   ollama pull llama3
        3. Ollama runs a local server automatically on localhost:11434

    Requires: pip install ollama
    """
    import ollama

    user_prompt = build_prompt(query, retrieved_docs)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


def generate_answer_anthropic(query: str, retrieved_docs: List[Dict], api_key: str = None) -> str:
    """
    Generate a grounded answer using the Anthropic API.
    Requires: pip install anthropic
    Requires: ANTHROPIC_API_KEY environment variable or passed directly.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    user_prompt = build_prompt(query, retrieved_docs)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def generate_answer_mock(query: str, retrieved_docs: List[Dict]) -> str:
    """
    A free, offline mock generator for testing the pipeline without API
    calls. Just returns the top retrieved document's content directly --
    this is NOT a real LLM response, it's a placeholder that proves the
    retrieval -> context -> answer pipeline wiring works end to end
    before you spend any API credits.
    """
    relevant_docs = [d for d in retrieved_docs if d.get("similarity_score", 0) >= 0.15]
    if not relevant_docs:
        return "I don't have data on that in this analysis."

    top_doc = relevant_docs[0]
    return (
        f"[MOCK ANSWER -- replace with real LLM call]\n"
        f"Based on {top_doc['source_notebook']}: {top_doc['content']}"
    )


def answer_question(query: str, retriever, backend: str = "mock") -> Dict:
    """
    Full RAG pipeline: retrieve -> generate -> return answer with sources.

    backend options:
        "mock"      - free, offline, returns retrieved text directly (no real LLM)
        "ollama"    - free, offline, real LLM running locally (requires Ollama installed)
        "anthropic" - paid API, best quality, requires ANTHROPIC_API_KEY
    """
    retrieved = retriever.retrieve(query, top_k=3)

    if backend == "mock":
        answer = generate_answer_mock(query, retrieved)
    elif backend == "ollama":
        answer = generate_answer_ollama(query, retrieved)
    elif backend == "anthropic":
        answer = generate_answer_anthropic(query, retrieved)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'mock', 'ollama', or 'anthropic'.")

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {"id": d["id"], "notebook": d["source_notebook"], "score": d.get("similarity_score")}
            for d in retrieved
        ],
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from retriever import get_retriever

    retriever = get_retriever("tfidf")

    test_queries = [
        "What's the win rate when a team secures Baron?",
        "Is Yasuo statistically overpowered?",
        "What's the weather like in Summoner's Rift?",
    ]

    for q in test_queries:
        result = answer_question(q, retriever, backend="mock")
        print(f"\nQ: {result['query']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
