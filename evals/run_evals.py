"""Eval harness for the Librarian.

Runs a golden set of question -> expected-source pairs and reports, per case,
whether retrieval surfaced the right document, plus overall hit rate and MRR.

Run:  python evals/run_evals.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from librarian.retrieve import retrieve  # noqa: E402

GOLDEN = [
    ("How much does the AI Sales Agent cost?", "pricing.md"),
    ("What's the price of the Growth package?", "pricing.md"),
    ("How much is Social Media Management per month?", "pricing.md"),
    ("What is the primary leak for a gym?", "targeting.md"),
    ("What is the primary leak for a roofing company?", "targeting.md"),
    ("Which businesses are on the blacklist?", "targeting.md"),
    ("What's the reply word prospects should send?", "outreach-sequence.md"),
    ("What is the cadence of the outreach sequence?", "outreach-sequence.md"),
    ("What tone should emails be written in?", "brand-voice.md"),
    ("Why are email scripts delivered as HTML files?", "brand-voice.md"),
    ("What services does TampifyAI offer?", "services.md"),
    ("What does the AI Lead Qualifier Agent do?", "services.md"),
]


def hit_at(question: str, expected: str, k: int) -> tuple[bool, int | None]:
    results = retrieve(question, k=k)
    sources = [r.source for r in results]
    for rank, src in enumerate(sources, start=1):
        if src == expected:
            return True, rank
    return False, None


def main():
    K = 4
    hits = 0
    mrr = 0.0
    misses = []

    print(f"Running {len(GOLDEN)} eval cases (top-{K} retrieval)\n")
    print(f"{'PASS':<5} {'RANK':<5} QUESTION")
    print("-" * 70)

    for question, expected in GOLDEN:
        ok, rank = hit_at(question, expected, K)
        if ok:
            hits += 1
            mrr += 1.0 / rank
        else:
            misses.append((question, expected))
        print(f"{'ok' if ok else 'MISS':<5} {str(rank or '-'):<5} {question}")

    n = len(GOLDEN)
    print("-" * 70)
    print(f"Hit@{K}:  {hits}/{n}  ({hits / n:.0%})")
    print(f"MRR:    {mrr / n:.3f}")

    if misses:
        print("\nMisses (retrieval did not surface the right file):")
        for q, exp in misses:
            print(f"  - {q}  (expected {exp})")
    return 0 if hits == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
