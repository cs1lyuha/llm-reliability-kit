import pytest

from llmkit.providers.fake import ScriptedProvider, text
from llmkit.types import LLMRequest, RetryableError, Usage


def req(msg="salut"):
    return LLMRequest(messages=[{"role": "user", "content": msg}])


def test_scripted_returns_steps_in_order():
    p = ScriptedProvider([text("unu"), text("doi")])
    assert p.complete(req()).text == "unu"
    assert p.complete(req()).text == "doi"
    assert p.calls == 2


def test_scripted_raises_errors():
    p = ScriptedProvider([RetryableError("429")])
    with pytest.raises(RetryableError):
        p.complete(req())


def test_scripted_callable_step_sees_request():
    p = ScriptedProvider([lambda r: text(r.messages[0]["content"].upper())])
    assert p.complete(req("abc")).text == "ABC"


def test_default_step_repeats():
    p = ScriptedProvider(default=text("mereu"))
    assert [p.complete(req()).text for _ in range(3)] == ["mereu"] * 3


def test_stream_yields_words():
    p = ScriptedProvider([text("a b c")])
    assert "".join(p.stream(req())).split() == ["a", "b", "c"]


def test_usage_addition():
    total = Usage(10, 2, 1, 0) + Usage(5, 3, 0, 4)
    assert (total.input_tokens, total.output_tokens, total.cache_read_tokens, total.cache_write_tokens) == (15, 5, 1, 4)
