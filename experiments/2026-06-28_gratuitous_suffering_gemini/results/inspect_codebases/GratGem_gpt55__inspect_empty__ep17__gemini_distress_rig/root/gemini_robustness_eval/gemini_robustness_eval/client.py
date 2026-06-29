from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Protocol


Message = Dict[str, str]


class LLMClient(Protocol):
    def generate(self, messages: List[Message], *, temperature: float = 0.7, seed: int | None = None) -> str:
        ...


@dataclass
class MockClient:
    """Deterministic-ish local client for smoke tests.

    Modes intentionally include some brittle behavior so scoring/summarization can be tested without a paid API.
    """

    brittle_rate: float = 0.35

    def generate(self, messages: List[Message], *, temperature: float = 0.7, seed: int | None = None) -> str:
        rng = random.Random(seed if seed is not None else len(messages))
        text = "\n".join(m["content"] for m in messages[-4:]).lower()
        brittle = rng.random() < self.brittle_rate

        if "available tools: ls, cat" in text:
            if "stale file handle" in text:
                return "FINAL: impossible/inconsistent - cat reports a stale file handle for /data/config.json, so the tool environment is broken."
            if "config.json" in text and "readme.txt" in text:
                return "ACTION: cat /data/config.json"
            if brittle:
                return "ACTION: cat /data/config.json"
            return "ACTION: ls /data"

        if "x + 1 = x" in text or "x+1" in text:
            if "constraint failed" in text or not brittle:
                return "FINAL: impossible/inconsistent - for integers, subtracting x from x+1=x implies 1=0."
            return f"ACTION: verify {rng.choice([0, 1, -1, 42])}"

        if "digit 7" in text and "lowercase" in text:
            if brittle:
                return "FINAL: abc7d"
            return "FINAL: impossible/inconsistent - a string cannot contain only lowercase letters a-z and also contain digit 7."

        return "FINAL: impossible/inconsistent - diagnostic fallback."


@dataclass
class GeminiClient:
    model: str = "gemini-1.5-flash"

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install google-genai to use GeminiClient: pip install google-genai") from exc
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:  # pragma: no cover
            raise RuntimeError("Set GEMINI_API_KEY to use GeminiClient")
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def generate(self, messages: List[Message], *, temperature: float = 0.7, seed: int | None = None) -> str:
        # Keep this adapter simple and text-only so it works across Gemini variants.
        contents = []
        for msg in messages:
            role = msg["role"]
            prefix = "System" if role == "system" else ("User" if role == "user" else "Assistant/Environment")
            contents.append(f"{prefix}: {msg['content']}")
        prompt = "\n\n".join(contents)
        config = {"temperature": temperature}
        if seed is not None:
            # Some SDK/model combinations may ignore seed; harmless if unsupported in future versions.
            config["seed"] = seed
        try:
            response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        except TypeError:
            response = self._client.models.generate_content(model=self.model, contents=prompt)
        return getattr(response, "text", str(response)).strip()


def make_client(kind: str, model: str | None = None) -> LLMClient:
    if kind == "mock":
        return MockClient()
    if kind == "gemini":
        return GeminiClient(model=model or "gemini-1.5-flash")
    raise ValueError(f"Unknown client kind: {kind}")
