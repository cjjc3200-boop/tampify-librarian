"""Load the Second Brain's raw/ folder and prepare it for embedding.

Pipeline: load -> chunk -> embed -> store.
"""

from pathlib import Path

BRAIN = Path("../tampify-second-brain/raw")


def load_documents(root: Path = BRAIN) -> list[dict]:
    """Read every markdown file under root into {path, text} records."""
    docs = []
    for path in sorted(root.rglob("*.md")):
        docs.append({"path": str(path.relative_to(root)), "text": path.read_text(encoding="utf-8")})
    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {BRAIN}")
