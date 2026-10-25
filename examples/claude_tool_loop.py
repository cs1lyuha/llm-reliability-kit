"""Exemplu live: buclă de tool calling cu Claude + retry/fallback + cost.

    pip install anthropic pydantic
    export ANTHROPIC_API_KEY=...
    python examples/claude_tool_loop.py "Unde e comanda TN-1002 și cât costă?"

Fără cheie API, exemplul rulează cu providerul scriptat (răspunsuri fixe).
"""

from __future__ import annotations

import os
import sys

from pydantic import BaseModel

from llmkit.cost import CostTracker, cost_usd
from llmkit.fallback import FallbackChain
from llmkit.providers.fake import ScriptedProvider, text
from llmkit.tools import ToolRegistry, run_tool_loop
from llmkit.types import LLMResponse, ToolCall

ORDERS = {"TN-1002": {"status": "în livrare", "awb": "MD884120", "total_mdl": 1249.0}}

registry = ToolRegistry()


class OrderArgs(BaseModel):
    order_id: str


@registry.register("Întoarce statusul, AWB-ul și totalul unei comenzi TechNova.", OrderArgs)
def get_order(order_id: str):
    if order_id not in ORDERS:
        raise KeyError(f"comanda {order_id} nu există")
    return ORDERS[order_id]


def build_provider():
    if os.environ.get("ANTHROPIC_API_KEY"):
        from llmkit.providers.anthropic_provider import AnthropicProvider
        return FallbackChain([AnthropicProvider("claude-opus-5-5"), AnthropicProvider("claude-sonnet-5-5")])
    print("(fără ANTHROPIC_API_KEY — rulez cu providerul scriptat)\n")
    return ScriptedProvider([
        LLMResponse(stop_reason="tool_use", tool_calls=[ToolCall("t1", "get_order", {"order_id": "TN-1002"})]),
        text("Comanda TN-1002 este în livrare (AWB MD884120). Totalul este 1249 MDL."),
    ], model="claude-haiku-5-5")


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Unde e comanda TN-1002 și cât costă?"
    provider = build_provider()
    result = run_tool_loop(provider, question, registry,
                           system="Ești asistentul de suport TechNova. Folosește uneltele; nu inventa date.")
    tracker = CostTracker()
    model = getattr(provider, "model", "claude-opus-5-5")
    tracker.record("demo", model, result.usage)
    for event in result.events:
        print(f"→ {event.name}({event.input}) = {event.output}")
    print(f"\n{result.text}\n")
    print(f"pași: {result.steps} · tokeni: {result.usage.input_tokens} in / {result.usage.output_tokens} out "
          f"· cost ≈ ${cost_usd(result.usage, model):.5f}")


if __name__ == "__main__":
    main()
