"""Load markdown from the Second Brain and split it into retrieval-sized chunks."""

from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Chunk:
    text: str
    source: str   # relative path within raw/
    index: int    # position within its source file


def load_documents(root: Path = None) -> list[tuple[str, str]]:
    """Return (relative_path, text) for every markdown file under root."""
    root = root or config.RAW_DIR
    docs = []
    for path in sorted(root.rglob("*.md")):
        rel = str(path.relative_to(root))
        docs.append((rel, path.read_text(encoding="utf-8")))
    return docs


def chunk_text(text: str, size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping windows, preferring paragraph boundaries."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(para), size - overlap):
                chunks.append(para[i : i + size].strip())
            continue

        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip()

    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def build_chunks(root: Path = None) -> list[Chunk]:
    """Load every document and chunk it, keeping source metadata."""
    chunks: list[Chunk] = []
    for rel, text in load_documents(root):
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(text=piece, source=rel, index=i))
    return chunks
