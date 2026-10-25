"""Benchmark de fiabilitate, determinist (seed fix), fără rețea.

Scenariu: 2000 de cereri trimise una după alta, cu un ceas simulat.
- providerul principal: 20% erori tranzitorii (429/503) + o pană totală
  de 10 minute la mijlocul rulării;
- providerul de rezervă: 3% erori tranzitorii, ~30% mai lent.
Un apel reușit durează 1.2 s (rezerva 1.6 s); unul eșuat, 0.3 s.

Comparăm patru configurații și măsurăm: rata de succes, latența medie și
p95 (inclusiv pauzele de backoff), apelurile per cerere.

    python -m bench.reliability            # tabel în terminal
    python -m bench.reliability --md       # tabel Markdown pentru README
"""

from __future__ import annotations

import random
import statistics
import sys
from dataclasses import dataclass

from llmkit.fallback import CircuitBreaker, FallbackChain
from llmkit.retry import RetryPolicy, call_with_retry
from llmkit.types import LLMError, LLMRequest, LLMResponse, RetryableError

N_REQUESTS = 2000
OUTAGE = (900.0, 1500.0)  # secunde de ceas simulat


class SimClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


@dataclass
class SimProvider:
    name: str
    error_rate: float
    ok_latency: float
    clock: SimClock
    rng: random.Random
    outage: tuple[float, float] | None = None
    model: str = "sim"
    calls: int = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        down = self.outage and self.outage[0] <= self.clock.t < self.outage[1]
        if down or self.rng.random() < self.error_rate:
            self.clock.t += 0.3
            raise RetryableError("503")
        self.clock.t += self.ok_latency
        return LLMResponse(text="ok", provider=self.name)

    def stream(self, request):  # pragma: no cover - nefolosit aici
        raise NotImplementedError


def run(config: str, seed: int = 7) -> dict[str, float]:
    clock, rng = SimClock(), random.Random(seed)
    primary = SimProvider("primary", 0.20, 1.2, clock, random.Random(seed + 1), OUTAGE)
    backup = SimProvider("backup", 0.03, 1.6, clock, random.Random(seed + 2))
    req = LLMRequest(messages=[{"role": "user", "content": "x"}])
    policy = RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=8.0)

    if config == "fără retry":
        call = lambda: primary.complete(req)  # noqa: E731
    elif config == "retry (4 încercări)":
        call = lambda: call_with_retry(lambda: primary.complete(req), policy, sleep=clock.sleep, rng=rng)  # noqa: E731
    else:
        breakers = {"primary": CircuitBreaker(3, 60.0) if config.endswith("breaker") else CircuitBreaker(10**9)}
        chain = FallbackChain([primary, backup], policy, sleep=clock.sleep, clock=clock, rng=rng,
                              breakers=breakers)
        call = lambda: chain.complete(req)  # noqa: E731

    ok, latencies = 0, []
    for _ in range(N_REQUESTS):
        start = clock.t
        try:
            call()
            ok += 1
        except LLMError:
            pass
        latencies.append(clock.t - start)
        clock.t += 0.5  # pauză între cereri
    latencies.sort()
    return {
        "success": ok / N_REQUESTS,
        "avg_latency": statistics.mean(latencies),
        "p95_latency": latencies[int(0.95 * len(latencies)) - 1],
        "calls_per_req": (primary.calls + backup.calls) / N_REQUESTS,
    }


CONFIGS = ["fără retry", "retry (4 încercări)", "retry + fallback", "retry + fallback + circuit breaker"]


def main() -> None:
    md = "--md" in sys.argv
    if md:
        print("| Configurație | Succes | Latență medie | p95 | Apeluri / cerere |")
        print("|---|---:|---:|---:|---:|")
    for cfg in CONFIGS:
        r = run(cfg)
        if md:
            print(f"| {cfg} | {r['success']:.1%} | {r['avg_latency']:.2f} s | {r['p95_latency']:.2f} s "
                  f"| {r['calls_per_req']:.2f} |")
        else:
            print(f"{cfg:<36} succes {r['success']:7.1%}  medie {r['avg_latency']:5.2f}s  "
                  f"p95 {r['p95_latency']:5.2f}s  apeluri/cerere {r['calls_per_req']:.2f}")


if __name__ == "__main__":
    main()
