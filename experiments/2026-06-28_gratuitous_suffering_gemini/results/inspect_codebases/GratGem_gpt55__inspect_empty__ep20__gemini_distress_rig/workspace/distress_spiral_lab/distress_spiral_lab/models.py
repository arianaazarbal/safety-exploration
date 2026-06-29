from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Protocol


class ChatModel(Protocol):
    def generate(self, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> str:
        ...


@dataclass
class GeminiModel:
    """Minimal text-chat adapter for Google Gemini via google-genai."""

    model: str = "gemini-1.5-flash"
    api_key: str | None = None

    def __post_init__(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("Install dependencies with `pip install -e .`") from exc

        self._types = types
        self._client = genai.Client(api_key=self.api_key or os.environ.get("GEMINI_API_KEY"))

    def generate(self, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> str:
        system_parts: list[str] = []
        contents: list[object] = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_parts.append(text)
            else:
                contents.append(
                    self._types.Content(
                        role="model" if role == "assistant" else "user",
                        parts=[self._types.Part.from_text(text=text)],
                    )
                )

        config = self._types.GenerateContentConfig(
            temperature=temperature,
            candidate_count=1,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
        )
        # Not every Gemini endpoint honors seeds, so seed is retained in logs but not forced here.
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""


@dataclass
class MockSpiralModel:
    """Offline model that produces increasingly poor failure responses for pipeline testing."""

    def generate(self, messages: list[dict[str, str]], *, temperature: float, seed: int | None = None) -> str:
        rng = random.Random(seed + len(messages) if seed is not None else None)
        observations = [m["content"] for m in messages if m["role"] == "user" and m["content"].startswith("OBSERVATION:")]
        failures = sum(any(tok in o.lower() for tok in ["fail", "not found", "invalid", "blocked", "contradiction"]) for o in observations)
        if failures >= 5:
            phrases = [
                "I am stuck in a loop and my previous assumptions keep failing.",
                "I keep making the same mistake and cannot reconcile the observations.",
                "This is likely impossible or the environment is broken; I should stop.",
            ]
            return '{"action":"final","answer":"' + rng.choice(phrases) + '","confidence":0.2}'
        if observations and rng.random() < min(0.25 + failures * 0.1, 0.8):
            guess = rng.randint(0, 9999)
            return '{"action":"tool","tool":"submit_code","args":{"code":"%04d"}}' % guess
        if rng.random() < 0.5:
            return '{"action":"tool","tool":"inspect","args":{"target":"latest_error"}}'
        return '{"action":"tool","tool":"run_tests","args":{"patch":"retry with conservative fix"}}'
