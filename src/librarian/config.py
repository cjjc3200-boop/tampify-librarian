"""Configuration for the TampifyAI Librarian."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Where the Second Brain lives (override with BRAIN_REPO_PATH for another checkout)
BRAIN_ROOT = Path(os.getenv("BRAIN_REPO_PATH", str(ROOT.parent / "tampify-second-brain")))
RAW_DIR = BRAIN_ROOT / "raw"

# Vector store location
CHROMA_DIR = Path(os.getenv("CHROMA_DB_PATH", str(ROOT / "data" / "chroma")))
COLLECTION = "tampify_brain"

# Embedding model (local, runs offline)
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# Chunking (character-based; ~4 chars/token, so 1200 chars ~= 300 tokens)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
TOP_K = int(os.getenv("TOP_K", "4"))

# Optional LLM generation. If unset, the Librarian answers extractively
# (returns the best-matching passages) so it works with zero API keys.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
