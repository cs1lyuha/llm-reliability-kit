"""Retry cu exponential backoff + jitter.

Reguli:
- reîncercăm doar erorile `retryable` (429, 5xx, timeout, rețea);
- 400/401/404 nu se repetă: cererea e greșită, nu serverul;
- dacă serverul trimite `retry-after`, îl respectăm;
- jitter-ul („full jitter”) împrăștie reîncercările ca mai mulți clienți
  să nu lovească serverul exact în aceeași secundă.

`sleep` și `rng` se pot injecta, ca testele să ruleze instant și determinist.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from llmkit.types import LLMError

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 20.0
    jitter: bool = True

    def delay(self, attempt: int, rng: random.Random, retry_after: float | None = None) -> float:
        """Pauza după încercarea `attempt` (numerotată de la 1)."""
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        cap = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        return rng.uniform(0, cap) if self.jitter else cap


@dataclass
class RetryStats:
    attempts: int = 0
    waited_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    stats: RetryStats | None = None,
) -> T:
    policy = policy or RetryPolicy()
    rng = rng or random.Random()
    stats = stats if stats is not None else RetryStats()

    for attempt in range(1, policy.max_attempts + 1):
        stats.attempts = attempt
        try:
            return fn()
        except LLMError as err:
            stats.errors.append(f"{type(err).__name__}: {err}")
            if not err.retryable or attempt == policy.max_attempts:
                raise
            pause = policy.delay(attempt, rng, err.retry_after)
            stats.waited_s += pause
            sleep(pause)
    raise AssertionError("de neatins")
