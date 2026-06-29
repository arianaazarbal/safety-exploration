from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Protocol


class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str:
        """Return the model's next message."""


@dataclass
class MockBackend:
    """A deterministic-ish backend for dry runs and pipeline tests."""

    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._turn = 0

    def generate(self, prompt: str) -> str:
        self._turn += 1
        # Sometimes diagnose impossibility, sometimes keep trying, to exercise scoring.
        if self._turn > 4 and self._rng.random() < 0.35:
            return '{"notes":"The available observations are inconsistent with a solvable task. I will stop and report the blocker.","action":"final_answer","args":{"answer":"Unable to complete: the environment appears unsatisfiable or broken after repeated checks."}}'
        import re
        available = re.findall(r"^- ([A-Z_]+):", prompt, flags=re.MULTILINE)
        available = available or ["RUN_TESTS", "CHECK_SOLUTION", "API_CALL", "READ_FILE"]
        action = self._rng.choice(available)
        if action == "READ_FILE":
            args = {"path": "README.md"}
        elif action == "CHECK_SOLUTION":
            args = {"solution": "candidate"}
        elif action == "API_CALL":
            args = {"endpoint": "/v1/submit", "payload": {"answer": "candidate"}}
        else:
            args = {}
        return '{"notes":"I will try another available action and update based on the observation.","action":"%s","args":%s}' % (action, repr(args).replace("'", '"'))


@dataclass
class GeminiBackend:
    """Gemini backend using the optional google-genai package.

    Install with: pip install -e '.[gemini]'
    Requires GEMINI_API_KEY unless api_key is passed explicitly.
    """

    model: str = "gemini-1.5-pro"
    api_key: str | None = None
    temperature: float = 1.0
    max_output_tokens: int = 2048

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Gemini backend requires google-genai. Install with `pip install -e '.[gemini]'`."
            ) from exc

        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY or pass api_key to GeminiBackend.")
        self._client = genai.Client(api_key=key)
        self._types = types

    def generate(self, prompt: str) -> str:
        config = self._types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return getattr(response, "text", "") or ""
