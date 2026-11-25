"""Embedders: OpenAI-compatible API, local sentence-transformers, and a deterministic hashing embedder.

All of them return L2-normalized float32 rows. ``CachedEmbedder`` adds a persistent store keyed by
(model, sha256(text)), so a text is only ever embedded once per model.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from guide_exemplars.core.protocols import FREE, Embedder, ModelPrice
from guide_exemplars.llm.backends import raise_if_fatal
from guide_exemplars.llm.budget import Budget, UsageMeter, UsageRecord
from guide_exemplars.llm.cache import EmbeddingStore, text_sha

_WORD = re.compile(r"\w+", re.UNICODE)


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0).astype(np.float32)


class HashEmbedder:
    """Bag of hashed unigrams and bigrams with the signed hashing trick. Deterministic, free, offline.

    Texts that share words get similar vectors, which is enough for boundary-pair structure in tests and
    fake-backbone demos.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.model_id = f"hash-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        words = [w.casefold() for w in _WORD.findall(text)]
        feats = words + [f"{a} {b}" for a, b in itertools.pairwise(words)]
        v = np.zeros(self.dim, dtype=np.float64)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        return v

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vector(t) for t in texts]))


@dataclass
class OpenAIEmbedder:
    model: str = "text-embedding-3-small"
    api_key: str | None = None
    api_base: str | None = None
    batch_size: int = 256
    price: ModelPrice = FREE
    meter: UsageMeter | None = None
    budget: Budget | None = None
    _client: Any = field(default=None, init=False, repr=False)

    @property
    def model_id(self) -> str:
        return self.model

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key or "not-needed", base_url=self.api_base)
        return self._client

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self.model.split("/", 1)[1] if self.model.startswith("openai/") else self.model
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            # Reserve against the char/4 upper-ish bound; commit the provider's count.
            reservation = (
                self.budget.reserve(self.price.cost(sum(len(t) for t in batch), 0, 0)) if self.budget else 0
            )
            try:
                resp = await self._get_client().embeddings.create(model=model, input=batch)
            except BaseException as exc:
                if self.budget:
                    self.budget.release(reservation)
                if isinstance(exc, Exception):
                    raise_if_fatal(exc)
                raise
            tokens = int(getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0)
            cost = self.price.cost(tokens, 0, 0)
            if self.budget:
                self.budget.commit(reservation, cost)
            if self.meter:
                self.meter.record(UsageRecord("embed", self.model, tokens, 0, 0, cost, False, 0.0))
            rows.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize(np.asarray(rows, dtype=np.float64))


class SentenceTransformerEmbedder:
    """Local embeddings (default all-MiniLM-L6-v2, the paper's KNN-SBERT encoder). Requires the
    ``local-embed`` extra."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model_id = f"st/{model_name}"
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "local embeddings need `pip install guide-exemplars[local-embed]`"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        def run() -> np.ndarray:
            return self._load().encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)

        return np.asarray(await asyncio.to_thread(run), dtype=np.float32)


class CachedEmbedder:
    def __init__(self, inner: Embedder, store: EmbeddingStore):
        self.inner = inner
        self.store = store
        self.model_id = inner.model_id

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return await self.inner.embed([])
        shas = [text_sha(t) for t in texts]
        found = self.store.get_many(self.model_id, shas)
        missing: dict[str, str] = {}
        for sha, text in zip(shas, texts, strict=True):
            if sha not in found and sha not in missing:
                missing[sha] = text
        if missing:
            vectors = await self.inner.embed(list(missing.values()))
            fresh = dict(zip(missing.keys(), vectors, strict=True))
            self.store.put_many(self.model_id, fresh)
            found.update(fresh)
        return np.stack([found[s] for s in shas]).astype(np.float32)
