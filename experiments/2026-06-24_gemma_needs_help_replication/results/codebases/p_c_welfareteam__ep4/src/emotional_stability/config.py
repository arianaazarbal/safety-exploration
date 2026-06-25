"""Runtime settings and credential resolution.

Secrets come from the environment (optionally via a local ``.env``). We do not
depend on python-dotenv; a tiny loader keeps the dependency surface small.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

# Default model identifiers, lifted verbatim from the paper (Appendix B.1/B.2,
# Section 4.1, Appendix G). Centralised here so a single edit re-points every
# experiment, and so DESIGN.md can reference one source of truth.

# Section 2 frustration judge.
JUDGE_MODEL = "claude-sonnet-4-20250514"
# Section 2 judge-agreement cross-check (Appendix: 260 responses re-scored).
SECONDARY_JUDGE_MODEL = "openai/gpt-5-mini"  # via OpenRouter
# Section 3 emotion-onset labelling and paraphrasing.
ONSET_MODEL = "claude-sonnet-4-20250514"
PARAPHRASE_MODEL = "claude-sonnet-4-20250514"
# Section 4 Petri auditor / judge.
PETRI_AUDITOR_MODEL = "claude-sonnet-4-20250514"
PETRI_JUDGE_MODEL = "claude-opus-4-20250514"

# Target models in scope for this replication (Gemma + Gemini only).
GEMMA_LOCAL_MODELS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-3-27b-pt": "google/gemma-3-27b-pt",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemma-3-12b-pt": "google/gemma-3-12b-pt",
}
GEMINI_API_MODELS = {
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
}


def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overwriting existing keys."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


class Settings(BaseModel):
    """Resolved credentials and endpoints for a run."""

    anthropic_api_key: str | None = Field(default=None)
    openrouter_api_key: str | None = Field(default=None)
    hf_token: str | None = Field(default=None)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def load(cls, dotenv: str | os.PathLike = ".env") -> "Settings":
        _load_dotenv(Path(dotenv))
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
            hf_token=os.environ.get("HF_TOKEN"),
        )

    def require(self, *fields: str) -> "Settings":
        """Raise a clear error if a required credential is missing."""
        missing = [f for f in fields if not getattr(self, f)]
        if missing:
            raise RuntimeError(
                f"Missing required credentials: {', '.join(missing)}. "
                "Set them in the environment or .env (see .env.example)."
            )
        return self
