import pytest

from llmkit.fallback import AllProvidersFailed, CircuitBreaker, FallbackChain
from llmkit.providers.fake import ScriptedProvider, text
from llmkit.retry import RetryPolicy
from llmkit.types import FatalError, LLMRequest, RefusalError, RetryableError

REQ = LLMRequest(messages=[{"role": "user", "content": "x"}])


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def chain(*providers, clock=None, attempts=2):
    return FallbackChain(list(providers), RetryPolicy(max_attempts=attempts), sleep=lambda _: None,
                         clock=clock or Clock())


def test_primary_answers_no_fallback():
    a, b = ScriptedProvider([text("A")], name="a"), ScriptedProvider([], name="b")
    c = chain(a, b)
    assert c.complete(REQ).provider == "a"
    assert b.calls == 0


def test_falls_back_when_primary_keeps_failing():
    a = ScriptedProvider(default=RetryableError("503"), name="a")
    b = ScriptedProvider([text("B")], name="b")
    c = chain(a, b)
    out = c.complete(REQ)
    assert out.provider == "b"
    assert a.calls == 2  # retry-ul a fost consumat înainte de fallback
    assert [t.ok for t in c.last_trace] == [False, True]


def test_fatal_error_does_not_fall_back():
    a = ScriptedProvider([FatalError("400")], name="a")
    b = ScriptedProvider([text("B")], name="b")
    with pytest.raises(FatalError):
        chain(a, b).complete(REQ)
    assert b.calls == 0


def test_refusal_falls_back_without_opening_circuit():
    a = ScriptedProvider([RefusalError("refusal")], name="a")
    b = ScriptedProvider([text("B")], name="b")
    c = chain(a, b)
    assert c.complete(REQ).provider == "b"
    assert c.breakers["a"].failures == 0


def test_all_fail_raises():
    a = ScriptedProvider(default=RetryableError("500"), name="a")
    b = ScriptedProvider(default=RetryableError("500"), name="b")
    with pytest.raises(AllProvidersFailed):
        chain(a, b).complete(REQ)


def test_circuit_opens_then_half_opens_after_cooldown():
    clock = Clock()
    a = ScriptedProvider(default=RetryableError("503"), name="a")
    b = ScriptedProvider(default=text("B"), name="b")
    c = chain(a, b, clock=clock, attempts=1)
    for _ in range(3):
        c.complete(REQ)
    assert c.breakers["a"].state == "open"
    calls_before = a.calls
    c.complete(REQ)
    assert a.calls == calls_before  # sărit
    assert c.last_trace[0].skipped
    clock.t = 31
    c.complete(REQ)
    assert a.calls == calls_before + 1  # cererea de probă


def test_breaker_resets_on_success():
    br = CircuitBreaker(failure_threshold=2)
    br.record_failure(0)
    br.record_success()
    br.record_failure(1)
    assert br.state == "closed"
