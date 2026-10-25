import random

import pytest

from llmkit.providers.fake import ScriptedProvider, text
from llmkit.retry import RetryPolicy, RetryStats, call_with_retry
from llmkit.types import FatalError, LLMRequest, RetryableError

REQ = LLMRequest(messages=[{"role": "user", "content": "x"}])


def no_sleep(_):
    pass


def test_succeeds_after_transient_errors():
    p = ScriptedProvider([RetryableError("503"), RetryableError("429"), text("ok")])
    stats = RetryStats()
    out = call_with_retry(lambda: p.complete(REQ), RetryPolicy(max_attempts=4), sleep=no_sleep, stats=stats)
    assert out.text == "ok"
    assert stats.attempts == 3
    assert len(stats.errors) == 2


def test_fatal_error_is_not_retried():
    p = ScriptedProvider([FatalError("400 bad request"), text("never")])
    with pytest.raises(FatalError):
        call_with_retry(lambda: p.complete(REQ), sleep=no_sleep)
    assert p.calls == 1


def test_gives_up_after_max_attempts():
    p = ScriptedProvider(default=RetryableError("500"))
    with pytest.raises(RetryableError):
        call_with_retry(lambda: p.complete(REQ), RetryPolicy(max_attempts=3), sleep=no_sleep)
    assert p.calls == 3


def test_retry_after_header_is_respected():
    slept = []
    p = ScriptedProvider([RetryableError("429", retry_after=7.0), text("ok")])
    call_with_retry(lambda: p.complete(REQ), sleep=slept.append)
    assert slept == [7.0]


@pytest.mark.parametrize("attempt,cap", [(1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (10, 20.0)])
def test_backoff_without_jitter_doubles_and_caps(attempt, cap):
    policy = RetryPolicy(base_delay=0.5, max_delay=20.0, jitter=False)
    assert policy.delay(attempt, random.Random(0)) == cap


def test_full_jitter_stays_under_cap():
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0)
    rng = random.Random(42)
    delays = [policy.delay(4, rng) for _ in range(200)]
    assert all(0 <= d <= 8.0 for d in delays)
    assert max(delays) - min(delays) > 4  # chiar sunt împrăștiate
