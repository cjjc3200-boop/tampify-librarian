# TampifyAI AI Librarian

A RAG system that answers questions over TampifyAI's own knowledge base (the [Second Brain](https://github.com/cjjc3200-boop/tampify-second-brain)).

This is not a demo over toy data — it indexes the real services, pricing, outreach sequence, targeting rules, and brand voice, and answers questions about them.

## Why it exists

The business keeps its knowledge in one place (the brain). This layer makes that knowledge answerable: instead of scrolling through markdown files, you ask a question and get a grounded answer with sources. It is also the retrieval layer that the Agency OS will draw on.

## Architecture

```
raw/ (Second Brain)  ->  ingest  ->  chunk  ->  embed  ->  Chroma
                                                             |
                              question  ->  embed  ->  hybrid retrieve (top-k)
                                                             |
                                              grounded answer + source files
```

### Retrieval is hybrid, and here is why

A pure vector search fuses meaning well but blurs exact tokens — prices (`$1,000`), proper nouns (`Gro Yoga`). Keyword search nails those but misses paraphrase. This system runs both and fuses them with **Reciprocal Rank Fusion (RRF)**, so the two rankings do not need to share a score scale.

### Embeddings run locally

`sentence-transformers` (`all-MiniLM-L6-v2`) runs on-device. No API key needed to index or retrieve — anyone who clones this can run it offline.

### Answers work with or without an LLM

- With `OPENAI_API_KEY` set: a grounded LLM answer, instructed to use only the retrieved context.
- Without it: an extractive fallback that returns the best-matching passages with their sources. The system is fully functional with zero API keys.

## Eval results

Retrieval quality is measured, not assumed. `evals/run_evals.py` runs a 12-question golden set (question -> the file that must be retrieved):

```
Hit@4:  12/12  (100%)
MRR:    0.917
```

11 of 12 questions retrieved the correct source at **rank 1**.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# point at the brain (defaults to ../tampify-second-brain)
export BRAIN_REPO_PATH=../tampify-second-brain

python src/cli.py index
python src/cli.py ask "How much is the AI Sales Agent?"

python evals/run_evals.py   # run the eval harness
```

## Layout

```
src/
  cli.py            # index | ask
  librarian/
    config.py       # paths, chunking, model, top-k
    ingest.py       # load + chunk
    store.py        # embed + persist to Chroma
    retrieve.py     # hybrid retrieval (vector + keyword, RRF)
    answer.py       # grounded generation + extractive fallback
evals/
  run_evals.py      # golden-set retrieval evals
```

## Stack

Python · Chroma · sentence-transformers · (optional) OpenAI

## Status

Working end to end: index -> retrieve -> answer, with evals passing. Next: FastAPI endpoint, tracing, and hybrid-weight tuning.
