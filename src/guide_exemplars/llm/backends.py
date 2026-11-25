"""Chat-completion backends. Each turns an ``LLMRequest`` into text plus token usage.

``OpenAIChatBackend`` is the default. It covers OpenAI itself and any OpenAI-compatible server (vLLM, SGLang,
llama.cpp, Ollama, OpenRouter, Groq, Together, ...) through ``api_base``. ``LiteLLMBackend`` is an optional
adapter for providers without an OpenAI-compatible endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from guide_exemplars.core.protocols import LLMRequest, ProviderFatal


@dataclass(frozen=True)
class BackendResponse:
    text: str
    prompt_tokens: int
    cached_prompt_tokens: int
    completion_tokens: int


class ChatBackend(Protocol):
    async def complete(self, req: LLMRequest, temperature: float) -> BackendResponse: ...


_FATAL_ERRORS = {"AuthenticationError", "PermissionDeniedError", "NotFoundError"}
_FATAL_CODES = {"insufficient_quota", "billing_hard_limit_reached", "invalid_api_key", "model_not_found"}


def raise_if_fatal(exc: Exception) -> None:
    """Turn account-level provider errors into ``ProviderFatal`` (matched by name, so no SDK import is needed).
    Ordinary rate limits and timeouts are left alone; the SDK retries those."""
    body = getattr(exc, "body", None)
    code = getattr(exc, "code", None) or (body.get("code") if isinstance(body, dict) else None)
    if type(exc).__name__ in _FATAL_ERRORS or code in _FATAL_CODES or "insufficient_quota" in str(exc):
        raise ProviderFatal(f"{type(exc).__name__}: {exc}") from exc


def _usage_from(resp: Any) -> tuple[int, int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0, 0
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return prompt, min(cached, prompt), completion


def _text_from(resp: Any) -> str:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    return (getattr(message, "content", None) or "") if message is not None else ""


@dataclass
class OpenAIChatBackend:
    model: str
    api_key: str | None = None
    api_base: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 4
    extra_body: dict[str, Any] = field(default_factory=dict)
    max_tokens_param: str = "max_tokens"
    json_mode_supported: bool = True
    _client: Any = field(default=None, init=False, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key or "not-needed",
                base_url=self.api_base,
                timeout=self.timeout_s,
                max_retries=self.max_retries,
            )
        return self._client

    def request_kwargs(self, req: LLMRequest, temperature: float) -> dict[str, Any]:
        model = self.model.split("/", 1)[1] if self.model.startswith("openai/") else self.model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": req.as_openai_messages(),
            "temperature": temperature,
            self.max_tokens_param: req.max_tokens,
        }
        if req.json_mode and self.json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs

    async def complete(self, req: LLMRequest, temperature: float) -> BackendResponse:
        try:
            resp = await self._get_client().chat.completions.create(**self.request_kwargs(req, temperature))
        except Exception as exc:
            raise_if_fatal(exc)
            raise
        prompt, cached, completion = _usage_from(resp)
        return BackendResponse(_text_from(resp), prompt, cached, completion)


@dataclass
class LiteLLMBackend:
    model: str
    api_key: str | None = None
    api_base: str | None = None
    timeout_s: float = 120.0
    max_retries: int = 4
    extra_body: dict[str, Any] = field(default_factory=dict)
    json_mode_supported: bool = True

    async def complete(self, req: LLMRequest, temperature: float) -> BackendResponse:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("LiteLLM backend requires `pip install guide-exemplars[litellm]`") from exc
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": req.as_openai_messages(),
            "temperature": temperature,
            "max_tokens": req.max_tokens,
            "timeout": self.timeout_s,
            "num_retries": self.max_retries,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if req.json_mode and self.json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        try:
            resp = await litellm.acompletion(**kwargs)
        except Exception as exc:
            raise_if_fatal(exc)
            raise
        prompt, cached, completion = _usage_from(resp)
        return BackendResponse(_text_from(resp), prompt, cached, completion)
