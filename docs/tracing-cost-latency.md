# Tracing, Latency & Cost

Measured behaviour of the Librarian, produced by `evals/benchmark.py` and the
`trace` command. These are real numbers from a run over the 5-file brain, not
estimates.

## Where the time goes

`python src/cli.py trace "How much is the AI Voice Agent?"` prints a span tree:

```
answer          9057.6 ms  question=How much is the AI Voice Agent?
  retrieve        9057.2 ms  k=4
    embed_query     7192.4 ms
    vector_search      8.7 ms  pool=12
    rrf_fuse           0.0 ms  candidates=10, kept=4
  generate           0.0 ms  model=extractive, mode=no_llm, ...
```

**The headline finding: the first query is slow because the embedding model
loads cold.** Everything after that is fast.

| Stage | Cold | Warm |
|---|---|---|
| embed_query | ~7,200 ms | < 20 ms |
| vector_search | 8.7 ms | ~5 ms |
| rrf_fuse | ~0 ms | ~0 ms |

## Steady-state latency

After a warm-up call -- which the benchmark does, and which a long-running
server does once at startup -- retrieval and answering are milliseconds:

```
Retrieval only (ms)
  p50: 19.6   p95: 28.7   min: 17.5   max: 31.4

End-to-end answer (ms)
  p50: 20.4   p95: 23.8
```

## Cost

Embeddings run locally (`sentence-transformers`) and are **free**. Retrieval
is a local vector lookup -- **free**. The only metered step is the optional LLM
generation, and only when `OPENAI_API_KEY` is set.

Estimated LLM cost per query (gpt-4o-mini rates, ~1,044 input / ~995 output
tokens for a typical retrieval):

```
mean: $0.000749   max: $0.000862
```

Roughly **$0.0007-0.0009 per question** -- about 1,300 questions per dollar.
Without an API key, the extractive path costs **$0**.

## Why this matters (the tradeoff story)

Three decisions follow directly from these numbers:

1. **Keep embeddings local.** A hosted embedding API would add a network round
trip and a per-call cost to every query. Local embeddings make retrieval both
faster in steady state and free. The price is a one-time model load -- paid
once at server startup, not per request.

2. **Warm the model at startup.** The cold-start number (9 s) is unacceptable
for a served API. The app loads the model on startup so user-facing requests
only ever see the warm (~20 ms) path.

3. **The LLM step is the only variable cost, so it is the only lever.**
Because retrieval is free and fast, cost control reduces to: how many tokens
do we feed the model, and how long is its output? That is why `TOP_K` and
`MAX_OUTPUT_TOKENS` are the two knobs that matter for spend.

## Reproduce

```bash
python src/cli.py trace "How much is the AI Voice Agent?"
python evals/benchmark.py
```
