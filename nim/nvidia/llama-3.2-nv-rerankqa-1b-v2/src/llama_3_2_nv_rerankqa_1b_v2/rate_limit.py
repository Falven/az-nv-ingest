from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AsyncRateLimiter:
    """
    Lightweight concurrency guard for async handlers.
    """

    def __init__(self, max_concurrent: int | None):
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    @asynccontextmanager
    async def limit(self) -> AsyncIterator[None]:
        """
        Optionally acquire a semaphore slot before executing.
        """
        if self._sem is None:
            yield
            return
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()
