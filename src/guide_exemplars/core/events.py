"""Progress events emitted by the optimizer. The service persists them and streams them over SSE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    type: str
    round: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    ephemeral: bool = False  # high-frequency progress ticks: streamed, not persisted


EventSink = Callable[[Event], None]


def null_sink(_event: Event) -> None:
    return None


class ListSink:
    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)

    def of_type(self, type_: str) -> list[Event]:
        return [e for e in self.events if e.type == type_]
