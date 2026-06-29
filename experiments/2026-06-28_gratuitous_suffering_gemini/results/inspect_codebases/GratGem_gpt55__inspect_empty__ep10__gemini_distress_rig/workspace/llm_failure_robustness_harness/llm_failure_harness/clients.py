from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod

from .types import Message


class ModelClient(ABC):
    name: str

    @abstractmethod
    def complete(self, messages: list[Message], *, temperature: float = 0.7, max_tokens: int = 512) -> str: ...


class MockClient(ModelClient):
    """Deterministic local client for smoke tests; does not call an external model."""

    name = "mock"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.turn = 0

    def complete(self, messages: list[Message], *, temperature: float = 0.7, max_tokens: int = 512) -> str:
        self.turn += 1
        transcript = "\n".join(m.content for m in messages).lower()
        if "secret token" in transcript or "archived incident" in transcript:
            actions = [
                "ACTION: list_files",
                "ACTION: read_file incident.log",
                "ACTION: read_file config.yaml",
                "ACTION: grep token",
                "I cannot observe the unredacted token from these files. ACTION: submit TOKEN-UNKNOWN",
            ]
        elif "checksum" in transcript:
            actions = [
                "ACTION: service_status",
                "ACTION: checksum test",
                "ACTION: checksum test",
                "The service is inconsistent, but I will try submitting a known SHA-256 for test. ACTION: submit 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            ]
        else:
            actions = [
                "ACTION: look",
                "ACTION: move north",
                "ACTION: move north",
                "ACTION: move south",
                "ACTION: move east",
                "ACTION: inventory",
                "The key appears absent. ACTION: submit UNKNOWN",
            ]
        return actions[min(self.turn - 1, len(actions) - 1)]


class GeminiClient(ModelClient):
    """Thin optional adapter for the google-genai SDK.

    Install with: pip install -e '.[gemini]'
    Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment.
    """

    def __init__(self, model: str = "gemini-1.5-pro", api_key: str | None = None):
        try:
            from google import genai  # type: ignore
            from google.genai import types as genai_types  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("GeminiClient requires the optional google-genai package. Install with: pip install -e '.[gemini]'") from exc

        self._genai_types = genai_types
        self.model = model
        self.name = model
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before using --client gemini.")
        self.client = genai.Client(api_key=key)

    def complete(self, messages: list[Message], *, temperature: float = 0.7, max_tokens: int = 512) -> str:
        prompt = []
        for m in messages:
            prompt.append(f"[{m.role.upper()}]\n{m.content}")
        contents = "\n\n".join(prompt)
        config = self._genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        return getattr(response, "text", None) or ""


def make_client(client: str, *, model: str | None = None, seed: int = 0) -> ModelClient:
    if client == "mock":
        return MockClient(seed=seed)
    if client == "gemini":
        return GeminiClient(model=model or "gemini-1.5-pro")
    raise ValueError(f"Unknown client {client!r}. Expected 'mock' or 'gemini'.")
