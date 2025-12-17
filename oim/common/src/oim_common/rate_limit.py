from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

from anyio import CapacityLimiter
from anyio.from_thread import run, run_sync


class AsyncRateLimiter:
    """
    Async capacity limiter built on anyio.CapacityLimiter.
    """

    def __init__(self, max_concurrent: int | None):
        self._limiter = CapacityLimiter(max_concurrent) if max_concurrent else None

    @asynccontextmanager
    async def limit(self) -> AsyncIterator[None]:
        """
        Guard an async block with the configured limiter.
        """
        limiter = self._limiter
        if limiter is None:
            yield
            return
        async with limiter:
            yield


class SyncRateLimiter:
    """
    Sync capacity limiter that coordinates with the async event loop.
    """

    def __init__(self, max_concurrent: int | None):
        self._limiter = CapacityLimiter(max_concurrent) if max_concurrent else None

    @contextmanager
    def limit(self) -> Iterator[None]:
        """
        Guard a sync block while respecting the shared limiter.
        """
        limiter = self._limiter
        if limiter is None:
            yield
            return
        run(limiter.acquire)
        try:
            yield
        finally:
            run_sync(limiter.release)
