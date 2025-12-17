Coordination rules for Triton refactors (applies to all agents)
===============================================================

- Edit scope is limited to `open-nv-ingest/nim/**`. Do not touch files outside this tree.
- `nim/baidu/paddleocr/**` is read-only and should be used only as a structural reference.
- Every NIM must be Triton-first: inference must run via NVIDIA Triton Inference Server and the official `tritonclient` (or Triton Python backend). No hand-rolled inference engines or custom protocol shims remain.
- Remove dead code and obsolete config after refactors. Keep `server.py`/`app.py` thin; prefer small, typed modules (settings, models, inference, metrics, auth, errors).
- Run `python -m ruff format` and `python -m ruff check` on any touched NIM/common modules. Add full type hints and docstrings to modified code.
