"""Embed chunks locally and persist them in Chroma."""

from functools import lru_cache

from . import config
from .ingest import Chunk, build_chunks


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings into normalized vectors."""
    vectors = _model().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def _client():
    import chromadb
    from chromadb.config import Settings

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def index(chunks: list[Chunk] | None = None, reset: bool = True) -> int:
    """Embed all chunks and write them to Chroma. Returns chunk count."""
    chunks = chunks if chunks is not None else build_chunks()
    if not chunks:
        raise RuntimeError(
            f"No documents found in {config.RAW_DIR}. Add markdown files to the brain first."
        )

    client = _client()
    if reset:
        try:
            client.delete_collection(config.COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=config.COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[f"{c.source}::{c.index}" for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        embeddings=embed([c.text for c in chunks]),
    )
    return len(chunks)


def collection():
    """Return the existing collection (must be indexed first)."""
    return _client().get_collection(config.COLLECTION)
