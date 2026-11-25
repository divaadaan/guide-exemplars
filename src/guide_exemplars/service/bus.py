"""In-process fan-out of job events to live SSE subscribers.

Workers run in threads with their own event loops, so publishing hops onto each subscriber's loop with
``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import threading
from collections import defaultdict
from typing import Any


class Broadcaster:
    def __init__(self) -> None:
        self._subs: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._subs[job_id].append((asyncio.get_running_loop(), queue))
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subs[job_id] = [(lp, q) for lp, q in self._subs[job_id] if q is not queue]
            if not self._subs[job_id]:
                del self._subs[job_id]

    def publish(self, job_id: str, message: dict[str, Any]) -> None:
        with self._lock:
            targets = list(self._subs.get(job_id, ()))
        for loop, queue in targets:

            def put(q: asyncio.Queue = queue) -> None:
                if not q.full():  # slow consumers drop ephemeral ticks rather than block the worker
                    q.put_nowait(message)

            try:
                loop.call_soon_threadsafe(put)
            except RuntimeError:  # loop closed
                self.unsubscribe(job_id, queue)
