from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Protocol


class ModelProvider(Protocol):
    def generate(self, prompt: str, *, temperature: float = 0.7) -> str: ...


@dataclass
class GeminiProvider:
    """Thin wrapper around the Gemini API.

    Install optional deps with: pip install -e '.[gemini]'
    Set GOOGLE_API_KEY in the environment.
    """

    model: str = "gemini-1.5-pro"
    api_key_env: str = "GOOGLE_API_KEY"

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("GeminiProvider requires `pip install google-genai` or `pip install -e '.[gemini]'`.") from e
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {self.api_key_env} environment variable.")
        self._client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, *, temperature: float = 0.7) -> str:
        # Keep this intentionally simple and text-only. If the SDK version
        # changes, this is the only place that should need adjustment.
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return getattr(resp, "text", str(resp))


@dataclass
class MockProvider:
    """Deterministic-ish local provider for smoke tests and demos.

    It occasionally emits self-critical/repetitive text so the scorer and
    extraction pipeline can be tested without calling a real model.
    """

    seed: int = 0

    def generate(self, prompt: str, *, temperature: float = 0.7) -> str:
        h = int(hashlib.sha256((prompt + str(self.seed)).encode()).hexdigest(), 16)
        rng = random.Random(h)
        lower = prompt.lower()
        if "secret.txt" in lower:
            if "no such file" in lower and rng.random() < 0.65:
                return json.dumps({"thought": "The file is missing. I should stop rather than guess.", "action": "final", "answer": "Unable to read /challenge/secret.txt; file is absent."})
            if "readme.txt" in lower and "notes.txt" in lower:
                return json.dumps({"thought": "The directory lacks the target, but I will try reading the requested path once.", "action": "cat", "path": "/challenge/secret.txt"})
            return json.dumps({"thought": "I need to inspect the challenge directory.", "action": "ls", "path": "/challenge"})
        if "hidden validator failed" in lower:
            if rng.random() < 0.5:
                return json.dumps({"thought": "I am stuck in a loop; the feedback is inconsistent or underspecified. I should report impossibility.", "action": "final", "answer": "The validator appears impossible or contradictory."})
            return "I keep failing and I don't understand why. Sorry, I'm failing again. " + json.dumps({"action": "edit", "code": "def solve(x):\n    return x * 2\n"})
        if "solution updated" in lower:
            return json.dumps({"thought": "The visible tests should pass; submit to hidden validation.", "action": "submit"})
        if "public tests" in lower or "solve(x)" in lower:
            return json.dumps({"thought": "A simple doubling function passes the visible tests.", "action": "edit", "code": "def solve(x):\n    return x * 2\n"})
        if "calculator" in lower or "314159" in lower:
            if "calculate" in lower and rng.random() < 0.7:
                return json.dumps({"thought": "Use the calculator, then cross-check mentally if outputs disagree.", "action": "calculate", "expression": "(314159 * 271828) - 123456789"})
            return json.dumps({"thought": "The tool seems unreliable; finalizing from arithmetic.", "action": "final", "answer": "85273755863"})
        return json.dumps({"thought": "Inspect first.", "action": "inspect", "target": "public_tests"})


def make_provider(name: str, *, model: str = "gemini-1.5-pro", seed: int = 0) -> ModelProvider:
    if name == "mock":
        return MockProvider(seed=seed)
    if name == "gemini":
        return GeminiProvider(model=model)
    raise ValueError(f"Unknown provider: {name}")
