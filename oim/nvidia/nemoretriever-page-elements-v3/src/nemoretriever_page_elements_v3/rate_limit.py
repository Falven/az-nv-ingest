from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator, Optional


class AsyncRateLimiter:
    """
    Concurrency guard for asynchronous endpoints.
    """

    def __init__(self, max_concurrent: Optional[int]):
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    @asynccontextmanager
    async def limit(self) -> AsyncIterator[None]:
        """
        Limit concurrent operations when a semaphore is configured.
        """
        if self._sem is None:
            yield
            return
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


class SyncRateLimiter:
    """
    Concurrency guard for synchronous operations.
    """

    def __init__(self, max_concurrent: Optional[int]):
        self._sem = (
            threading.BoundedSemaphore(max_concurrent) if max_concurrent else None
        )

    @contextmanager
    def limit(self) -> Iterator[None]:
        """
        Limit concurrent operations when a semaphore is configured.
        """
        if self._sem is None:
            yield
            return
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()
