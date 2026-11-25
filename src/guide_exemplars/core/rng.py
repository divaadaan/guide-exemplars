"""Named, derived random streams.

Each component draws from its own stream, keyed by a path such as ``("round", 2, "ops")``. Extra draws in one
component (a retried call, an extra candidate) never shift the randomness of another. A run resumed from a
checkpoint therefore continues exactly as an uninterrupted run would.
"""

from __future__ import annotations

import zlib

import numpy as np

_INT32_MAX = 2**31 - 1


def _path_key(path: tuple[str | int, ...]) -> int:
    return zlib.crc32("/".join(str(p) for p in path).encode("utf-8"))


class RngStreams:
    """Factory for independent, reproducible ``numpy`` generators derived from one seed."""

    def __init__(self, seed: int):
        self.seed = int(seed)

    def get(self, *path: str | int) -> np.random.Generator:
        """Fresh generator for ``path``. The same seed and path always give the same stream."""
        return np.random.default_rng([self.seed, _path_key(path)])

    def int_seed(self, *path: str | int) -> int:
        """Deterministic integer seed for libraries that take ``random_state`` (e.g. sklearn)."""
        return int(self.get(*path).integers(0, _INT32_MAX))

    def __repr__(self) -> str:
        return f"RngStreams(seed={self.seed})"
