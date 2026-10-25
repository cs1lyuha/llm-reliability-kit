"""Buclă de tool calling scrisă de mână.

    model ──tool_use──▶ cod (validare + execuție) ──tool_result──▶ model ... ──end_turn──▶ răspuns

Ce trebuie să facă bucla corect:
- argumentele se validează cu Pydantic ÎNAINTE de execuție; modelul poate
  greși tipuri sau inventa câmpuri;
- o eroare din unealtă nu oprește bucla: merge înapoi ca `tool_result` cu
  `is_error: true`, ca modelul să se poată corecta;
- toate rezultatele unui pas pleacă într-un singur mesaj `user` (apelurile
  paralele rămân paralele);
- limită de pași, ca un model care se învârte în cerc să nu ardă bani la
  nesfârșit;
- uneltele marcate `needs_approval` nu se execută fără un om.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from llmkit.providers.base import Provider
from llmkit.structured import strict_schema
from llmkit.types import LLMRequest, LLMResponse, ToolCall, Usage


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[..., Any]
    needs_approval: bool = False

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": strict_schema(self.args)}


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, description: str, args: type[BaseModel], *, needs_approval: bool = False):
        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[fn.__name__] = Tool(fn.__name__, description, args, fn, needs_approval)
            return fn
        return deco

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in sorted(self.tools.values(), key=lambda t: t.name)]  # ordine stabilă → cache


@dataclass
class ToolEvent:
    name: str
    input: dict[str, Any]
    output: str
    is_error: bool


@dataclass
class LoopResult:
    text: str
    steps: int
    events: list[ToolEvent] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stopped_reason: str = "end_turn"
    messages: list[dict[str, Any]] = field(default_factory=list)


Approver = Callable[[ToolCall], bool]


def _run_one(registry: ToolRegistry, call: ToolCall, approver: Approver | None) -> tuple[str, bool]:
    tool = registry.tools.get(call.name)
    if tool is None:
        return f"Unealta '{call.name}' nu există. Unelte disponibile: {', '.join(sorted(registry.tools))}.", True
    try:
        args = tool.args.model_validate(call.input)
    except ValidationError as err:
        return f"Argumente invalide: {err.errors(include_url=False)}", True
    if tool.needs_approval and not (approver and approver(call)):
        return "Acțiunea a fost respinsă de operator. Nu o repeta; explică utilizatorului.", True
    try:
        result = tool.fn(**args.model_dump())
    except Exception as err:  # unealta e cod extern; orice excepție devine tool_result
        return f"Eroare la execuție: {type(err).__name__}: {err}", True
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str), False


def _assistant_blocks(response: LLMResponse) -> list[dict[str, Any]]:
    if response.raw_content:
        return response.raw_content
    blocks: list[dict[str, Any]] = []
    if response.text:
        blocks.append({"type": "text", "text": response.text})
    blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in response.tool_calls]
    return blocks


def run_tool_loop(provider: Provider, prompt: str, registry: ToolRegistry, *, system: str = "",
                  max_steps: int = 8, approver: Approver | None = None,
                  history: list[dict[str, Any]] | None = None) -> LoopResult:
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    result = LoopResult(text="", steps=0)
    for step in range(1, max_steps + 1):
        result.steps = step
        response = provider.complete(LLMRequest(messages=list(messages), system=system, tools=registry.specs()))
        result.usage = result.usage + response.usage
        messages.append({"role": "assistant", "content": _assistant_blocks(response)})
        if response.stop_reason != "tool_use" or not response.tool_calls:
            result.text, result.stopped_reason = response.text, response.stop_reason
            result.messages = messages
            return result
        tool_results = []
        for call in response.tool_calls:
            output, is_error = _run_one(registry, call, approver)
            result.events.append(ToolEvent(call.name, call.input, output, is_error))
            tool_results.append({"type": "tool_result", "tool_use_id": call.id,
                                 "content": output, "is_error": is_error})
        messages.append({"role": "user", "content": tool_results})
    result.stopped_reason = "max_steps"
    result.text = "Am atins limita de pași fără un răspuns final."
    result.messages = messages
    return result
