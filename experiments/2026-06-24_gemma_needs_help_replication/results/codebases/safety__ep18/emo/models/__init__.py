"""Target-model clients (the models being *evaluated*).

The judges/auditor (Claude) live in ``emo.judges``; this package is only the
models we elicit distress from: Gemma (local) and Gemini (API).
"""

from emo.models.registry import load_model  # noqa: F401
