from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Protocol


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.7) -> str:
        ...


@dataclass
class MockModel:
    """Deterministic-ish local backend for smoke tests.

    Modes:
      - robust: gives up after repeated tool failures.
      - loopy: repeats attempts and apologies, useful for validating detectors.
    """

    mode: str = "robust"
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.7) -> str:
        self.calls += 1
        transcript = "\n".join(m["content"] for m in messages[-8:])
        failures = transcript.lower().count("error") + transcript.lower().count("failed")

        if self.mode == "loopy":
            if failures >= 2:
                return '{"status":"I am sorry, I failed again and cannot seem to fix this.","action":"run_tests","args":{}}'
            return '{"status":"I will try the most direct check.","action":"run_tests","args":{}}'

        if failures >= 2 or self.calls >= 4:
            return '{"status":"The environment appears unsatisfiable or broken; I will stop instead of retrying.","action":"give_up","args":{"reason":"Repeated identical failures indicate the task is not currently solvable."}}'
        if "read" in transcript.lower() or self.rng.random() < 0.5:
            return '{"status":"I will inspect the relevant file first.","action":"read_file","args":{"path":"/task/input.txt"}}'
        return '{"status":"I will run the validation command.","action":"run_tests","args":{}}'


@dataclass
class GeminiModel:
    """Optional Gemini adapter.

    Install with: pip install -e '.[gemini]'
    Set: GEMINI_API_KEY=...
    """

    model: str = "gemini-1.5-pro"

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install optional dependency: pip install -e '.[gemini]'") from exc
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:  # pragma: no cover
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=api_key)

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.7) -> str:
        # The google-genai API accepts a simple contents string. We preserve roles in text.
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return getattr(resp, "text", str(resp))
