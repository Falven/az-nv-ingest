from __future__ import annotations

import os

import uvicorn

from open_shims.app import create_app

app = create_app()


def run() -> None:
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("open_shims.main:app", host="0.0.0.0", port=port, reload=False, log_level="info")


if __name__ == "__main__":
    run()
