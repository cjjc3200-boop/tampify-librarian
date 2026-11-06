"""Latency + cost benchmark for the Librarian.

Runs the golden question set, traces each one, and reports latency
distribution (p50/p95) and estimated cost per query. Produces the raw
material for the cost/latency write-up.

Run:  python evals/benchmark.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from librarian import answer as answer_mod  # noqa: E402
from librarian.retrieve import retrieve  # noqa: E402

QUESTIONS = [
    "How much does the AI Sales Agent cost?",
    "What is the primary leak for a gym?",
    "What's the reply word prospects should send?",
    "What tone should emails be written in?",
    "What services does TampifyAI offer?",
    "Which businesses are on the blacklist?",
    "What is the Growth package?",
    "What is the primary leak for a roofing company?",
]

REPEATS = 3


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def main():
    print(f"Benchmarking {len(QUESTIONS)} questions x {REPEATS} repeats\n")

    # Warm up: load the embedding model so the first measured call is fair.
    retrieve("warmup")

    retrieval_times: list[float] = []
    end_to_end_times: list[float] = []
    costs: list[float] = []

    for q in QUESTIONS:
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            retrieve(q)
            retrieval_times.append((time.perf_counter() - t0) * 1000)

        result = answer_mod.answer(q)
        end_to_end_times.append(result["latency_ms"])
        costs.append(result["cost_usd"])

    rt = sorted(retrieval_times)
    e2e = sorted(end_to_end_times)

    print("Retrieval only (ms)")
    print(f"  p50: {statistics.median(rt):.1f}   p95: {percentile(rt, 95):.1f}   min: {min(rt):.1f}   max: {max(rt):.1f}")
    print("\nEnd-to-end answer (ms)")
    print(f"  p50: {statistics.median(e2e):.1f}   p95: {percentile(e2e, 95):.1f}")
    print("\nCost per query (USD)")
    print(f"  mean: ${statistics.mean(costs):.6f}   max: ${max(costs):.6f}")
    print("\n  Note: embeddings are local (free). LLM cost applies only when an")
    print("  API key is set; without one the extractive path is $0.")


if __name__ == "__main__":
    main()
