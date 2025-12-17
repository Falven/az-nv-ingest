from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AsyncRateLimiter:
    """
    Lightweight concurrency guard for async HTTP handlers.
    """

    def __init__(self, max_concurrent: int | None):
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    @asynccontextmanager
    async def limit(self) -> AsyncIterator[None]:
        """
        Acquire an optional semaphore before processing a request.
        """
        if not self._sem:
            yield
            return
        await self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()
