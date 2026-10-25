"""Provider fals, scriptat. Îl folosesc în teste, în demo și în benchmark.

Primește o listă de „pași”: fiecare pas e fie un LLMResponse (răspuns
reușit), fie o excepție (eroare simulată), fie o funcție care primește
cererea și întoarce unul dintre ele.
"""

from __future__ import annotations

from typing import Callable, Iterator, Union

from llmkit.types import LLMError, LLMRequest, LLMResponse, Usage

Step = Union[LLMResponse, LLMError, Callable[[LLMRequest], "LLMResponse | LLMError"]]


class ScriptedProvider:
    def __init__(self, steps: list[Step] | None = None, *, name: str = "fake",
                 model: str = "fake-model", default: Step | None = None):
        self.name = name
        self.model = model
        self._steps = list(steps or [])
        self._default = default
        self.requests: list[LLMRequest] = []

    def _next(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._steps:
            step = self._steps.pop(0)
        elif self._default is not None:
            step = self._default
        else:
            raise AssertionError(f"{self.name}: nu mai sunt pași în script")
        if callable(step) and not isinstance(step, (LLMResponse, LLMError)):
            step = step(request)
        if isinstance(step, LLMError):
            raise step
        step.provider = step.provider or self.name
        step.model = step.model or self.model
        return step

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self._next(request)

    def stream(self, request: LLMRequest) -> Iterator[str]:
        response = self._next(request)
        for word in response.text.split(" "):
            yield word + " "

    @property
    def calls(self) -> int:
        return len(self.requests)


def text(value: str, *, input_tokens: int = 100, output_tokens: int = 20) -> LLMResponse:
    """Scurtătură pentru teste: un răspuns text simplu."""
    return LLMResponse(text=value, usage=Usage(input_tokens, output_tokens))
