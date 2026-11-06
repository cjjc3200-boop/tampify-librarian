"""Produce an answer from retrieved chunks."""

from . import config
from .retrieve import Result, retrieve

_PROMPT = """You are the TampifyAI Librarian. Answer the question using ONLY the context below.
If the answer is not in the context, say you don't have that in the brain.
Be concise and quote exact numbers when relevant.

Context:
{context}

Question: {question}
Answer:"""


def _context(results: list[Result]) -> str:
    return "\n\n".join(f"[{r.source}]\n{r.text}" for r in results)


def _extractive(question: str, results: list[Result]) -> str:
    if not results:
        return "I don't have that in the brain yet."
    lines = ["(No LLM key set — showing the best-matching passages.)", ""]
    for r in results:
        lines.append(f"From {r.source}:\n{r.text}")
        lines.append("")
    return "\n".join(lines).strip()


def answer(question: str, k: int | None = None, results: list[Result] | None = None) -> dict:
    results = results if results is not None else retrieve(question, k=k)
    context = _context(results)

    if config.OPENAI_API_KEY:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": _PROMPT.format(context=context, question=question)}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
    else:
        text = _extractive(question, results)

    return {"answer": text, "sources": [r.source for r in results], "context": context}
