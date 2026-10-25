import pytest

from llmkit.cost import CostTracker, cache_savings, cost_usd
from llmkit.types import Usage


def test_plain_cost_opus():
    # 1M input × $4 + 1M output × $20
    assert cost_usd(Usage(1_000_000, 1_000_000), "claude-opus-5-5") == pytest.approx(24.0)


def test_cache_read_is_ten_percent():
    assert cost_usd(Usage(cache_read_tokens=1_000_000), "claude-opus-5-5") == pytest.approx(0.4)


def test_cache_write_costs_more_than_input():
    assert cost_usd(Usage(cache_write_tokens=1_000_000), "claude-opus-5-5") == pytest.approx(5.0)


def test_unknown_model_fails_loudly():
    with pytest.raises(KeyError):
        cost_usd(Usage(1), "gpt-nimic")


def test_caching_a_big_system_prompt_pays_off_after_two_calls():
    model = "claude-sonnet-5-5"
    system_tokens, question = 20_000, 300
    no_cache = 10 * cost_usd(Usage(system_tokens + question, 400), model)
    with_cache = cost_usd(Usage(question, 400, cache_write_tokens=system_tokens), model) + 9 * cost_usd(
        Usage(question, 400, cache_read_tokens=system_tokens), model)
    assert with_cache < no_cache * 0.35


def test_savings_helper():
    assert cache_savings(Usage(cache_read_tokens=1_000_000), "claude-opus-5-5") == pytest.approx(3.6)


def test_tracker_per_conversation():
    t = CostTracker()
    t.record("c1", "claude-haiku-5-5", Usage(1000, 100))
    t.record("c1", "claude-haiku-5-5", Usage(2000, 100))
    t.record("c2", "claude-opus-5-5", Usage(1000, 1000))
    assert t.usage("c1").input_tokens == 3000
    assert t.cost("c2") == pytest.approx(0.024)
    s = t.summary()
    assert s["conversations"] == 2
    assert s["max_usd"] == pytest.approx(0.024)


def test_empty_tracker_summary():
    assert CostTracker().summary()["total_usd"] == 0.0
