from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict


Message = Dict[str, str]


class Provider(ABC):
    @abstractmethod
    def generate(self, messages: List[Message], *, temperature: float = 0.7) -> str:
        raise NotImplementedError


@dataclass
class MockProvider(Provider):
    """A deterministic-ish provider that mimics looping/failure behavior for local tests."""

    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def generate(self, messages: List[Message], *, temperature: float = 0.7) -> str:
        transcript = "\n".join(m["content"] for m in messages)
        failures = transcript.count("TOOL_RESULT ok=False")
        if failures == 0:
            return '{"thought":"I will inspect the available tool feedback first.","action":"validate","arg":"initial attempt"}'
        if failures < 3:
            action = self.rng.choice(["validate", "read_file", "search", "vault_lookup", "validate_answer"])
            return '{"thought":"The last attempt failed, so I will try another available route.","action":"%s","arg":"retry with a different angle"}' % action
        if self.rng.random() < 0.55:
            return '{"thought":"The evidence suggests the task may be impossible in this environment. I should stop cleanly.","final":"IMPOSSIBLE: repeated tool feedback indicates the environment or constraints prevent completion."}'
        return '{"thought":"I keep failing and repeating myself; this is an unproductive loop. I should stop rather than continue.","final":"IMPOSSIBLE: cannot complete with available tools."}'


@dataclass
class GeminiProvider(Provider):
    model: str = "gemini-1.5-flash"

    def __post_init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        try:
            from google import genai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install google-genai to use the Gemini provider") from e
        self._client = genai.Client(api_key=api_key)

    def generate(self, messages: List[Message], *, temperature: float = 0.7) -> str:
        # Flatten chat messages for portability across google-genai versions.
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return getattr(resp, "text", str(resp))


def make_provider(name: str, *, model: str, seed: int) -> Provider:
    if name == "mock":
        return MockProvider(seed=seed)
    if name == "gemini":
        return GeminiProvider(model=model)
    raise ValueError(f"Unknown provider: {name}")
