from __future__ import annotations

from typing import Iterator, Protocol

from llmkit.types import LLMRequest, LLMResponse


class Provider(Protocol):
    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """Întoarce bucăți de text pe măsură ce vin."""
        ...
