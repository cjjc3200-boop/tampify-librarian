"""A tiny, dependency-free tracer for the Librarian pipeline.

Records a span per stage (embed, retrieve, generate) with duration, depth and
attributes, so you can see exactly what happened on a query and where the time
went. Deliberately self-contained -- no external tracing service or API key --
so the repo runs anywhere. Spans can be dumped to JSON or (optionally)
shipped to any backend later.

Usage:
    tracer = Tracer()
    with tracer.span("retrieve", query=q):
        ...
    print(tracer.tree())
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    depth: int = 0
    duration_ms: float = 0.0
    attributes: dict = field(default_factory=dict)


class Tracer:
    def __init__(self):
        self.spans: list[Span] = []
        self._depth = 0

    @contextmanager
    def span(self, name: str, **attributes):
        s = Span(name=name, depth=self._depth, attributes=dict(attributes))
        self.spans.append(s)
        self._depth += 1
        t0 = time.perf_counter()
        try:
            yield s
        finally:
            self._depth -= 1
            s.duration_ms = (time.perf_counter() - t0) * 1000.0

    def total_ms(self) -> float:
        """Wall time of the outermost span (not the sum of nested spans)."""
        return self.spans[0].duration_ms if self.spans else 0.0

    def by_name(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]

    def summary(self) -> dict:
        return {
            "total_ms": round(self.total_ms(), 2),
            "spans": [
                {
                    "name": s.name,
                    "depth": s.depth,
                    "duration_ms": round(s.duration_ms, 2),
                    "attributes": s.attributes,
                }
                for s in self.spans
            ],
        }

    def tree(self) -> str:
        lines = []
        for s in self.spans:
            pad = "  " * s.depth
            attrs = ""
            if s.attributes:
                attrs = "  " + json.dumps(s.attributes, default=str)
            lines.append(f"{pad}{s.name}  {s.duration_ms:.1f}ms{attrs}")
        return "\n".join(lines)

    def to_json(self, path=None) -> str:
        payload = json.dumps(self.summary(), indent=2, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        return payload
