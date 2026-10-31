# TampifyAI AI Librarian

The engine over the Second Brain. Ingests `raw/`, maintains `wiki/`, and answers questions about the business.

## What it does

1. **Ingest** — read every file in the brain's `raw/` folder
2. **Chunk** — split into retrieval-sized pieces with overlap
3. **Embed + store** — vectorize and persist to Chroma
4. **Summarize** — write the `wiki/` mirror layer
5. **Retrieve** — answer grounded questions over the knowledge

This is the same load → chunk → embed → store → retrieve → generate pipeline as my Vito's Sandwich Shop project, pointed at business knowledge instead of a restaurant FAQ.

## Depends on

- `tampify-second-brain` — the data it reads and maintains
- Consumed by `tampify-agency-os` — which uses retrieved knowledge to do work

## Stack

Python · LangChain · Chroma · python-dotenv

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill it in
```

## Status

Scaffolded. Pipeline implementation in progress.
