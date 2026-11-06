"""Command-line entry point: index the brain, ask questions, show traces."""

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
    print(f"Latency: {result['latency_ms']} ms   Cost: ${result['cost_usd']:.6f}")


def cmd_trace(args):
    result = answer_mod.answer(args.question, k=args.k)
    tracer_summary = result["trace"]
    print(f"Question: {args.question}\n")
    for s in tracer_summary["spans"]:
        pad = "  " * s["depth"]
        attrs = ""
        if s["attributes"]:
            attrs = "  " + ", ".join(f"{k}={v}" for k, v in s["attributes"].items())
        print(f"{pad}{s['name']:<14} {s['duration_ms']:>7.1f} ms{attrs}")
    print(f"\nTotal: {tracer_summary['total_ms']} ms   Cost: ${result['cost_usd']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="TampifyAI Librarian")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="embed the brain into Chroma")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=config.TOP_K)
    p_ask.set_defaults(func=cmd_ask)

    p_trace = sub.add_parser("trace", help="ask a question and show the trace tree")
    p_trace.add_argument("question")
    p_trace.add_argument("-k", type=int, default=config.TOP_K)
    p_trace.set_defaults(func=cmd_trace)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
