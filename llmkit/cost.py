"""Cost per request și per conversație.

Prețurile sunt în USD per 1 milion de tokeni (listă publică, octombrie 2026).
Tokenii din cache se plătesc diferit:
- scriere în cache: 1.25 × prețul de input;
- citire din cache: 0.1 × prețul de input.
Așa se vede în cifre de ce merită prompt caching pe un system prompt mare.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llmkit.types import Usage

# model: (input, output)  — USD / 1M tokeni
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "fake-model": (1.00, 5.00),
}
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10


def cost_usd(usage: Usage, model: str) -> float:
    if model not in PRICES:
        raise KeyError(f"preț necunoscut pentru modelul {model!r}")
    price_in, price_out = PRICES[model]
    total = (
        usage.input_tokens * price_in
        + usage.cache_write_tokens * price_in * CACHE_WRITE_MULT
        + usage.cache_read_tokens * price_in * CACHE_READ_MULT
        + usage.output_tokens * price_out
    )
    return total / 1_000_000


def cache_savings(usage: Usage, model: str) -> float:
    """Cât am fi plătit în plus dacă tokenii citiți din cache erau input normal."""
    price_in, _ = PRICES[model]
    return usage.cache_read_tokens * price_in * (1 - CACHE_READ_MULT) / 1_000_000


@dataclass
class CostTracker:
    """Adună consumul pe conversații. Un singur obiect per proces e suficient."""

    conversations: dict[str, list[tuple[str, Usage]]] = field(default_factory=dict)

    def record(self, conversation_id: str, model: str, usage: Usage) -> None:
        self.conversations.setdefault(conversation_id, []).append((model, usage))

    def cost(self, conversation_id: str) -> float:
        return sum(cost_usd(u, m) for m, u in self.conversations.get(conversation_id, []))

    def usage(self, conversation_id: str) -> Usage:
        total = Usage()
        for _, u in self.conversations.get(conversation_id, []):
            total = total + u
        return total

    def summary(self) -> dict[str, float]:
        costs = [self.cost(cid) for cid in self.conversations]
        if not costs:
            return {"conversations": 0, "total_usd": 0.0, "avg_usd": 0.0, "max_usd": 0.0}
        return {"conversations": len(costs), "total_usd": sum(costs),
                "avg_usd": sum(costs) / len(costs), "max_usd": max(costs)}
