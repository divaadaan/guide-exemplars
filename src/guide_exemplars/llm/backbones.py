"""Backbone configuration (``configs/backbones.yaml``) and the factories that turn it into clients."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from guide_exemplars.core.protocols import FREE, Embedder
from guide_exemplars.llm.backends import ChatBackend, LiteLLMBackend, OpenAIChatBackend
from guide_exemplars.llm.budget import Budget, UsageMeter
from guide_exemplars.llm.cache import (
    EmbeddingStore,
    MemoryEmbeddingStore,
    ResponseCache,
    SQLiteEmbeddingStore,
    SQLiteResponseCache,
)
from guide_exemplars.llm.client import LLMClient
from guide_exemplars.llm.embedders import (
    CachedEmbedder,
    HashEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
)
from guide_exemplars.llm.fake import FakeBackend, Responder
from guide_exemplars.llm.pricing import PriceTable
from guide_exemplars.llm.ratelimit import TokenRateLimiter
from guide_exemplars.paths import CACHE_DIR, CONFIG_DIR

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate(value: Any, env: dict[str, str] | None = None) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` in strings, recursively. An unset variable with no default
    becomes the empty string."""
    env = os.environ if env is None else env
    if isinstance(value, str):
        return _VAR.sub(lambda m: env.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: interpolate(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v, env) for v in value]
    return value


class BackboneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["openai", "litellm", "fake"] = "openai"
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    self_hosted: bool = False
    temperature: float = 0.2
    max_concurrency: int = Field(default=16, ge=1)
    timeout_s: float = 120.0
    max_retries: int = 4
    # Client-side pacing to stay under the provider's tokens-per-minute limit (prompt + max_tokens per call).
    # None = no pacing (self-hosted servers queue requests themselves).
    tpm_limit: int | None = Field(default=None, gt=0)
    json_mode: bool = True
    strip_think: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)
    max_tokens_param: str = "max_tokens"
    embedder: str = "openai/text-embedding-3-small"
    embedder_api_base: str | None = None
    embedder_api_key_env: str | None = "OPENAI_API_KEY"

    def api_key(self, override: str | None = None) -> str | None:
        if override:
            return override
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    def missing_settings(self, api_key_override: str | None = None) -> list[str]:
        """What prevents this backbone from running right now (empty list = ready)."""
        missing = []
        if self.kind != "fake":
            if self.self_hosted and not self.api_base:
                missing.append("api_base (set the env var referenced in backbones.yaml)")
            if not self.self_hosted and not self.api_key(api_key_override):
                missing.append(f"API key ({self.api_key_env})")
        if self.embedder.startswith("openai/") and not (
            api_key_override or os.environ.get(self.embedder_api_key_env or "")
        ):
            missing.append(f"embedder API key ({self.embedder_api_key_env})")
        return missing

    def redacted(self) -> dict[str, Any]:
        """Safe to persist: never contains keys (only env var names)."""
        return self.model_dump(mode="json")


def load_backbones(
    path: str | Path | None = None, env: dict[str, str] | None = None
) -> dict[str, BackboneConfig]:
    path = Path(path) if path else CONFIG_DIR / "backbones.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for name, spec in (data.get("backbones") or {}).items():
        spec = interpolate(spec, env)
        spec = {k: (None if v == "" else v) for k, v in spec.items()}
        out[name] = BackboneConfig(name=name, **spec)
    return out


def default_response_cache() -> ResponseCache:
    return SQLiteResponseCache(CACHE_DIR / "cache.db")


def default_embedding_store() -> EmbeddingStore:
    return SQLiteEmbeddingStore(CACHE_DIR / "cache.db")


_LIMITERS: dict[tuple[str, str, int], TokenRateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def shared_rate_limiter(cfg: BackboneConfig) -> TokenRateLimiter | None:
    """One limiter per (model, endpoint, limit) for the whole process: TPM is an account limit, and the benchmark
    and the service can build several clients for the same backbone."""
    if not cfg.tpm_limit or cfg.kind == "fake":
        return None
    key = (cfg.model, cfg.api_base or "", cfg.tpm_limit)
    with _LIMITERS_LOCK:
        if key not in _LIMITERS:
            _LIMITERS[key] = TokenRateLimiter(cfg.tpm_limit)
        return _LIMITERS[key]


def build_llm(
    cfg: BackboneConfig,
    *,
    cache: ResponseCache | None = None,
    meter: UsageMeter | None = None,
    budget: Budget | None = None,
    prices: PriceTable | None = None,
    api_key: str | None = None,
    responder: Responder | None = None,
    fake_failure_rate: float = 0.0,
    fake_jitter_s: float = 0.0,
    cache_only: bool = False,
) -> LLMClient:
    prices = prices or PriceTable.load()
    backend: ChatBackend
    if cfg.kind == "fake":
        backend = FakeBackend(
            **({"responder": responder} if responder else {}),
            failure_rate=fake_failure_rate,
            jitter_s=fake_jitter_s,
        )
    elif cfg.kind == "litellm":
        backend = LiteLLMBackend(
            model=cfg.model,
            api_key=cfg.api_key(api_key),
            api_base=cfg.api_base,
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
            extra_body=cfg.extra_body,
            json_mode_supported=cfg.json_mode,
        )
    else:
        backend = OpenAIChatBackend(
            model=cfg.model,
            api_key=cfg.api_key(api_key),
            api_base=cfg.api_base,
            timeout_s=cfg.timeout_s,
            max_retries=cfg.max_retries,
            extra_body=cfg.extra_body,
            max_tokens_param=cfg.max_tokens_param,
            json_mode_supported=cfg.json_mode,
        )
    price = FREE if cfg.self_hosted else prices.lookup(cfg.model)
    return LLMClient(
        backend,
        cfg.model,
        price=price,
        cache=cache,
        meter=meter,
        budget=budget,
        max_concurrency=cfg.max_concurrency,
        default_temperature=cfg.temperature,
        strip_reasoning=cfg.strip_think,
        cache_namespace=cfg.api_base or "",
        extra_cache_key=cfg.extra_body or None,
        cache_only=cache_only,
        rate_limiter=shared_rate_limiter(cfg),
    )


def build_embedder(
    spec: str,
    *,
    store: EmbeddingStore | None = None,
    meter: UsageMeter | None = None,
    budget: Budget | None = None,
    prices: PriceTable | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    api_key_env: str | None = "OPENAI_API_KEY",
) -> Embedder:
    """``hash`` | ``hash-<dim>`` | ``st/<sentence-transformers model>`` | ``openai/<embedding model>``."""
    inner: Embedder
    if spec == "hash" or spec.startswith("hash-"):
        dim = int(spec.split("-", 1)[1]) if "-" in spec else 256
        inner = HashEmbedder(dim)
    elif spec.startswith("st/"):
        inner = SentenceTransformerEmbedder(spec.split("/", 1)[1])
    elif spec.startswith("openai/"):
        prices = prices or PriceTable.load()
        inner = OpenAIEmbedder(
            model=spec,
            api_key=api_key or (os.environ.get(api_key_env) if api_key_env else None),
            api_base=api_base,
            price=prices.lookup(spec) or FREE,
            meter=meter,
            budget=budget,
        )
    else:
        raise ValueError(f"unknown embedder spec {spec!r}")
    return CachedEmbedder(inner, store if store is not None else MemoryEmbeddingStore())
