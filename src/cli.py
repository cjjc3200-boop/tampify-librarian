"""Command-line entry point: index the brain, then ask questions."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from librarian import answer as answer_mod  # noqa: E402
from librarian import config, store  # noqa: E402


def cmd_index(_args):
    from librarian.ingest import build_chunks

    chunks = build_chunks()
    n = store.index(chunks)
    files = sorted({c.source for c in chunks})
    print(f"Indexed {n} chunks from {len(files)} files:")
    for f in files:
        print(f"  - {f}")


def cmd_ask(args):
    result = answer_mod.answer(args.question, k=args.k)
    print(result["answer"])
    print("\nSources:", ", ".join(dict.fromkeys(result["sources"])))


def main():
    parser = argparse.ArgumentParser(description="TampifyAI Librarian")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="embed the brain into Chroma")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=config.TOP_K)
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
