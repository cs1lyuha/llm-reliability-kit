"""Tipurile comune: cerere, răspuns, apel de unealtă, consum de tokeni.

Sunt independente de provider, ca restul bibliotecii (retry, fallback,
structured outputs, tool loop) să nu știe dacă vorbește cu Claude,
cu un model local sau cu un provider fals din teste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    # blocurile brute, ca să le putem trimite înapoi neschimbate în istoric
    raw_content: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LLMRequest:
    messages: list[dict[str, Any]]
    system: str = ""
    max_tokens: int = 4096
    tools: list[dict[str, Any]] = field(default_factory=list)
    json_schema: dict[str, Any] | None = None
    cache_system: bool = False
    effort: str | None = None
    timeout_s: float = 60.0


class LLMError(Exception):
    """Eroare de bază. `retryable` spune dacă are sens să mai încercăm."""

    retryable = False

    def __init__(self, message: str, *, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class RetryableError(LLMError):
    """429, 5xx, timeout, conexiune căzută."""

    retryable = True


class FatalError(LLMError):
    """400, 401, 404: dacă repetăm cererea, primim aceeași eroare."""


class RefusalError(LLMError):
    """Modelul a refuzat cererea (stop_reason = refusal)."""
