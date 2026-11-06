"""Produce an answer from retrieved chunks.

If OPENAI_API_KEY is set, the Librarian answers with an LLM grounded in the
retrieved context. If not, it falls back to an extractive answer (the best
passages, with their sources) so the whole system works with zero API keys.

Every call is traced end to end and, when the LLM step runs, its token usage
and USD cost are recorded on the span and returned in the result.
"""

from . import config, cost
from .retrieve import Result, retrieve
from .tracing import Tracer

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
    lines = ["(No LLM key set -- showing the best-matching passages.)", ""]
    for r in results:
        lines.append(f"From {r.source}:\n{r.text}")
        lines.append("")
    return "\n".join(lines).strip()


def answer(
    question: str,
    k: int | None = None,
    results: list[Result] | None = None,
    tracer: Tracer | None = None,
) -> dict:
    """Answer a question, tracing every stage.

    Returns answer, sources, trace, usage (tokens), cost_usd and latency_ms.
    """
    own_tracer = tracer is None
    tracer = tracer or Tracer()

    def _run():
        if results is not None:
            retrieved = results
        else:
            with tracer.span("retrieve", k=k or config.TOP_K):
                retrieved = retrieve(question, k=k, tracer=tracer)
        context = _context(retrieved)

        usage = {"input_tokens": 0, "output_tokens": 0}
        prompt = _PROMPT.format(context=context, question=question)

        if config.OPENAI_API_KEY:
            with tracer.span("generate", model=config.LLM_MODEL):
                from openai import OpenAI

                client = OpenAI(api_key=config.OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=config.MAX_OUTPUT_TOKENS,
                )
                text = resp.choices[0].message.content.strip()
            # Prefer provider-reported usage; fall back to estimation.
            u = getattr(resp, "usage", None)
            usage["input_tokens"] = getattr(u, "prompt_tokens", 0) or cost.estimate_tokens(prompt)
            usage["output_tokens"] = getattr(u, "completion_tokens", 0) or cost.estimate_tokens(text)
        else:
            # Extractive path: no LLM, but estimate what the call WOULD cost.
            with tracer.span("generate", model="extractive", mode="no_llm"):
                text = _extractive(question, retrieved)
            usage["input_tokens"] = cost.estimate_tokens(prompt)
            usage["output_tokens"] = cost.estimate_tokens(text)

        usd = cost.llm_cost_usd(usage["input_tokens"], usage["output_tokens"])
        tracer.spans[-1].attributes.update(
            {
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "cost_usd": round(usd, 8),
            }
        )
        return text, context, retrieved, usage, usd

    if own_tracer:
        with tracer.span("answer", question=question):
            text, context, retrieved, usage, usd = _run()
    else:
        text, context, retrieved, usage, usd = _run()

    return {
        "answer": text,
        "sources": [r.source for r in retrieved],
        "context": context,
        "usage": usage,
        "cost_usd": round(usd, 8),
        "latency_ms": round(tracer.total_ms(), 2),
        "trace": tracer.summary(),
    }
