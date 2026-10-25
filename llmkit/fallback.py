"""Fallback între provideri + circuit breaker.

Ordinea e fixă: principalul, apoi rezervele. Fiecare provider are retry-ul
lui. Dacă un provider pică de `failure_threshold` ori la rând, circuitul lui
se „deschide” și e sărit `cooldown_s` secunde, ca să nu pierdem timp pe
retry-uri la un serviciu care e clar căzut. După cooldown primește din nou
o cerere de probă („half-open”).

Erorile fatale (400 etc.) NU declanșează fallback: cererea e greșită și ar
pica la fel și la al doilea provider. Refuzul modelului, în schimb, merge
mai departe — alt model poate răspunde.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from llmkit.providers.base import Provider
from llmkit.retry import RetryPolicy, RetryStats, call_with_retry
from llmkit.types import FatalError, LLMError, LLMRequest, LLMResponse, RefusalError


class AllProvidersFailed(LLMError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    cooldown_s: float = 30.0
    failures: int = 0
    opened_at: float | None = None

    def allow(self, now: float) -> bool:
        if self.opened_at is None:
            return True
        return now - self.opened_at >= self.cooldown_s  # half-open: o cerere de probă

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self, now: float) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = now

    @property
    def state(self) -> str:
        return "open" if self.opened_at is not None else "closed"


@dataclass
class Attempt:
    provider: str
    ok: bool
    attempts: int
    error: str = ""
    skipped: bool = False


@dataclass
class FallbackChain:
    providers: list[Provider]
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    rng: random.Random = field(default_factory=random.Random)
    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    last_trace: list[Attempt] = field(default_factory=list)

    def __post_init__(self) -> None:
        for p in self.providers:
            self.breakers.setdefault(p.name, CircuitBreaker())

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.last_trace = []
        last_error: LLMError | None = None
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not breaker.allow(self.clock()):
                self.last_trace.append(Attempt(provider.name, False, 0, "circuit open", skipped=True))
                continue
            stats = RetryStats()
            try:
                response = call_with_retry(
                    lambda: provider.complete(request), self.policy,
                    sleep=self.sleep, rng=self.rng, stats=stats,
                )
            except FatalError:
                raise
            except LLMError as err:
                if not isinstance(err, RefusalError):
                    breaker.record_failure(self.clock())
                self.last_trace.append(Attempt(provider.name, False, stats.attempts, str(err)))
                last_error = err
                continue
            breaker.record_success()
            self.last_trace.append(Attempt(provider.name, True, stats.attempts))
            return response
        raise AllProvidersFailed(f"niciun provider n-a răspuns: {last_error}")
