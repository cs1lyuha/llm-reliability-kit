"""Adaptor pentru API-ul Anthropic (Claude).

Traduce `LLMRequest` în parametrii SDK-ului și răspunsul înapoi în
`LLMResponse`. Retry-ul SDK-ului e oprit (`max_retries=0`): de retry și
fallback se ocupă `llmkit.retry` / `llmkit.fallback`, ca politica să fie
aceeași pentru toți providerii.
"""

from __future__ import annotations

from typing import Any, Iterator

from llmkit.types import (FatalError, LLMError, LLMRequest, LLMResponse, RefusalError, RetryableError,
                          ToolCall, Usage)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def classify_error(err: Exception) -> LLMError:
    import anthropic

    if isinstance(err, anthropic.APIConnectionError):  # include APITimeoutError
        return RetryableError(f"rețea/timeout: {err}")
    if isinstance(err, anthropic.APIStatusError):
        retry_after = err.response.headers.get("retry-after") if err.response is not None else None
        seconds = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
        if err.status_code in RETRYABLE_STATUS or err.status_code >= 500:
            return RetryableError(f"HTTP {err.status_code}", retry_after=seconds)
        return FatalError(f"HTTP {err.status_code}: {err.message}")
    return FatalError(f"{type(err).__name__}: {err}")


class AnthropicProvider:
    def __init__(self, model: str = "claude-opus-5-5", *, client: Any = None, server_fallback: bool = True):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.name = f"anthropic:{model}"
        # fallback pe server la refuz (beta); dacă nu e dorit, rămâne fallback-ul nostru din FallbackChain
        self.server_fallback = server_fallback

    def build_params(self, request: LLMRequest) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "max_tokens": request.max_tokens,
                                  "messages": request.messages}
        if request.system:
            if request.cache_system:
                params["system"] = [{"type": "text", "text": request.system,
                                     "cache_control": {"type": "ephemeral"}}]
            else:
                params["system"] = request.system
        if request.tools:
            params["tools"] = request.tools
        output_config: dict[str, Any] = {}
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if request.effort:
            output_config["effort"] = request.effort
        if output_config:
            params["output_config"] = output_config
        if self.server_fallback:
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
        return params

    def _messages_api(self, request: LLMRequest) -> Any:
        client = self.client.with_options(timeout=request.timeout_s, max_retries=0)
        return client.beta.messages if self.server_fallback else client.messages

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            message = self._messages_api(request).create(**self.build_params(request))
        except Exception as err:
            raise classify_error(err) from err
        return self.parse(message)

    def stream(self, request: LLMRequest) -> Iterator[str]:
        try:
            with self._messages_api(request).stream(**self.build_params(request)) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except Exception as err:
            raise classify_error(err) from err
        if final.stop_reason == "refusal":
            raise RefusalError("modelul a refuzat cererea")

    def parse(self, message: Any) -> LLMResponse:
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise RefusalError(f"refuz: {getattr(details, 'category', None)}")
        texts, calls, raw = [], [], []
        for block in message.content:
            raw.append(block.to_dict() if hasattr(block, "to_dict") else dict(block))
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input)))
        u = message.usage
        usage = Usage(u.input_tokens, u.output_tokens, getattr(u, "cache_read_input_tokens", 0) or 0,
                      getattr(u, "cache_creation_input_tokens", 0) or 0)
        return LLMResponse(text="".join(texts), tool_calls=calls, stop_reason=message.stop_reason,
                           usage=usage, model=message.model, provider=self.name, raw_content=raw)
