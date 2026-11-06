"""Token estimation and cost model.

Embedding is local (sentence-transformers) and therefore free. The only
metered cost is the optional LLM generation step. This module estimates token
usage and converts it to dollars using configurable per-million rates.

Rates are read from config so they can be updated without touching code --
verify them against your provider's current pricing page.
"""

from . import config


def estimate_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken when available, else ~4 chars/token."""
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def llm_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for one LLM call at the configured rates."""
    return (
        input_tokens / 1_000_000 * config.LLM_INPUT_COST_PER_1M
        + output_tokens / 1_000_000 * config.LLM_OUTPUT_COST_PER_1M
    )
