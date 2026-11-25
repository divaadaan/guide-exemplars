"""Token counting for budget reservations and cost estimates.

Uses tiktoken when available (``o200k_base``, the GPT-4o family encoding). Otherwise it falls back to a
~4-characters-per-token heuristic. That is close enough for estimates on non-OpenAI models, whose real usage is
read back from each response anyway.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _encoding(name: str):  # pragma: no cover - trivial wrapper
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return None


class TokenCounter:
    def __init__(self, encoding: str = "o200k_base"):
        self._enc = _encoding(encoding)

    @property
    def exact(self) -> bool:
        return self._enc is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        return max(1, round(len(text) / 4))

    def count_messages(self, messages: tuple[tuple[str, str], ...]) -> int:
        # ~4 tokens of chat-format overhead per message, plus reply priming.
        return sum(self.count(content) + 4 for _, content in messages) + 3
