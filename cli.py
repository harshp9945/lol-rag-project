"""Interactive CLI for the RAG assistant.

Usage:
    python cli.py                                   # tfidf + mock (offline demo)
    python cli.py --backend ollama                  # local llama3.1 via Ollama
    python cli.py --retriever embeddings --backend anthropic
    python cli.py --threshold 0.12                  # override refusal threshold
"""
import argparse
import logging
import sys

from rag_assistant.chunker import load_knowledge_base
from rag_assistant.llm.backends import AnthropicBackend, MockBackend, OllamaBackend
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.retrieval.tfidf import TfidfRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
log = logging.getLogger("cli")


def build(args) -> RAGPipeline:
    if args.retriever == "embeddings":
        from rag_assistant.retrieval.embeddings import EmbeddingsRetriever
        retriever = EmbeddingsRetriever()
    else:
        retriever = TfidfRetriever()

    backends = {"mock": MockBackend, "ollama": OllamaBackend, "anthropic": AnthropicBackend}
    try:
        backend = backends[args.backend]()
    except Exception as exc:
        log.error("could not start backend '%s': %s", args.backend, exc)
        log.error("falling back to mock backend so retrieval is still inspectable")
        backend = MockBackend()

    chunks = load_knowledge_base(args.kb_dir)
    if not chunks:
        log.error("knowledge base '%s' is empty; add .md files first", args.kb_dir)
        sys.exit(1)
    retriever.index(chunks)
    pipe = RAGPipeline(retriever, backend, threshold=args.threshold)
    log.info("ready: retriever=%s backend=%s chunks=%d threshold=%.3f",
             retriever.name, backend.name, len(chunks), pipe.threshold)
    return pipe


def main():
    parser = argparse.ArgumentParser(description="LoL analytics RAG assistant")
    parser.add_argument("--retriever", default="tfidf", choices=["tfidf", "embeddings"])
    parser.add_argument("--backend", default="mock", choices=["mock", "ollama", "anthropic"])
    parser.add_argument("--kb-dir", default="knowledge_base")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the per-backend refusal threshold")
    args = parser.parse_args()
    pipe = build(args)

    print("Ask about the capstone findings (Ctrl-D or 'quit' to exit).")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"quit", "exit"}:
            break
        resp = pipe.ask(q)
        tag = "REFUSED" if resp.refused else f"score={resp.top_score:.3f}"
        print(f"[{tag}] {resp.answer}")
        if not resp.refused:
            print("sources:", ", ".join(sorted({c.source for c in resp.chunks})))


if __name__ == "__main__":
    main()
