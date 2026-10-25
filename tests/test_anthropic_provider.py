"""Adaptorul Anthropic testat cu un client fals (fără rețea, fără cheie API)."""

from types import SimpleNamespace as NS

import anthropic
import httpx2
import pytest

from llmkit.providers.anthropic_provider import AnthropicProvider, classify_error
from llmkit.types import FatalError, LLMRequest, RefusalError, RetryableError

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def block(**kw):
    return NS(**kw, to_dict=lambda: dict(kw))


def message(content, stop_reason="end_turn", **usage):
    u = dict(input_tokens=10, output_tokens=5, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    u.update(usage)
    return NS(content=content, stop_reason=stop_reason, model="claude-opus-5-5", usage=NS(**u), stop_details=None)


class FakeMessages:
    def __init__(self, result):
        self.result = result
        self.params = None

    def create(self, **params):
        self.params = params
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result):
        self.messages = FakeMessages(result)
        self.beta = NS(messages=self.messages)
        self.options = None

    def with_options(self, **kw):
        self.options = kw
        return self


def provider(result, **kw):
    client = FakeClient(result)
    return AnthropicProvider(client=client, **kw), client


def test_text_response_is_parsed():
    p, _ = provider(message([block(type="text", text="Salut!")], cache_read_input_tokens=7))
    out = p.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert out.text == "Salut!"
    assert out.usage.cache_read_tokens == 7
    assert out.provider == "anthropic:claude-opus-5-5"


def test_tool_use_and_thinking_blocks_are_kept_raw():
    content = [block(type="thinking", thinking="", signature="sig"),
               block(type="tool_use", id="tu1", name="get_order", input={"order_id": "A1"})]
    p, _ = provider(message(content, stop_reason="tool_use"))
    out = p.complete(LLMRequest(messages=[]))
    assert out.tool_calls[0].name == "get_order"
    assert out.raw_content[0]["type"] == "thinking"  # se trimite înapoi neschimbat


def test_params_include_schema_effort_cache_and_fallback():
    p, client = provider(message([block(type="text", text="{}")]))
    p.complete(LLMRequest(messages=[], system="S" * 10, cache_system=True, json_schema={"type": "object"},
                          effort="low", timeout_s=12))
    params = client.messages.params
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}},
                                       "effort": "low"}
    assert params["fallbacks"] == "default"
    assert client.options == {"timeout": 12, "max_retries": 0}


def test_no_server_fallback_uses_plain_messages():
    p, client = provider(message([block(type="text", text="x")]), server_fallback=False)
    p.complete(LLMRequest(messages=[]))
    assert "fallbacks" not in client.messages.params


def test_refusal_raises():
    p, _ = provider(message([], stop_reason="refusal"))
    with pytest.raises(RefusalError):
        p.complete(LLMRequest(messages=[]))


@pytest.mark.parametrize("status,kind", [(429, RetryableError), (529, RetryableError), (500, RetryableError),
                                         (400, FatalError), (401, FatalError), (404, FatalError)])
def test_status_classification(status, kind):
    resp = httpx2.Response(status, headers={"retry-after": "2"}, request=REQUEST)
    err = anthropic.APIStatusError("err", response=resp, body=None)
    assert isinstance(classify_error(err), kind)


def test_retry_after_is_read():
    resp = httpx2.Response(429, headers={"retry-after": "3"}, request=REQUEST)
    p, _ = provider(anthropic.RateLimitError("rate", response=resp, body=None))
    with pytest.raises(RetryableError) as info:
        p.complete(LLMRequest(messages=[]))
    assert info.value.retry_after == 3.0


def test_timeout_is_retryable():
    assert isinstance(classify_error(anthropic.APITimeoutError(request=REQUEST)), RetryableError)
