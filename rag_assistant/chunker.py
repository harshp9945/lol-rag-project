"""Chunker: splits knowledge-base documents into retrievable pieces.

Retrieval works best when each indexed unit holds one coherent idea.
Whole documents are too broad; single sentences lose context. Strategy:
paragraph-based chunks with a word-count ceiling. Both knobs are
parameters so eval experiments can vary them.
"""
from pathlib import Path


def chunk_text(text: str, source: str, max_words: int = 120) -> list[dict]:
    """Split one document into chunks of at most max_words words."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        words = len(para.split())
        if current and current_words + words > max_words:
            chunks.append({"text": "\n\n".join(current), "source": source})
            current, current_words = [], 0
        current.append(para)
        current_words += words

    if current:
        chunks.append({"text": "\n\n".join(current), "source": source})
    return chunks


def load_knowledge_base(kb_dir: str, max_words: int = 120) -> list[dict]:
    """Read every .md/.txt file in kb_dir and return all chunks."""
    chunks: list[dict] = []
    for path in sorted(Path(kb_dir).glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            chunks.extend(chunk_text(path.read_text(encoding="utf-8"), path.name, max_words))
    return chunks
