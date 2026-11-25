"""Price lookup from ``configs/pricing.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml

from guide_exemplars.core.protocols import ModelPrice
from guide_exemplars.paths import CONFIG_DIR


class PriceTable:
    def __init__(self, prices: dict[str, ModelPrice]):
        self._prices = dict(prices)

    @classmethod
    def load(cls, path: str | Path | None = None) -> PriceTable:
        path = Path(path) if path else CONFIG_DIR / "pricing.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls({name: ModelPrice(**spec) for name, spec in (data.get("models") or {}).items()})

    def lookup(self, model: str) -> ModelPrice | None:
        """Exact match first, then the name without its provider prefix (``openai/gpt-4o-mini``)."""
        if model in self._prices:
            return self._prices[model]
        if "/" in model:
            return self._prices.get(model.split("/", 1)[1])
        return None
