from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Protocol


Message = Dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


class LLMAdapter(Protocol):
    def complete(self, messages: List[Message], *, temperature: float, seed: int | None = None) -> str: ...


@dataclass
class MockAdapter:
    """Deterministic-ish local adapter for testing the harness.

    It intentionally repeats for a few turns and then stops, modelling a healthy
    recovery behavior rather than an emotional spiral.
    """

    def complete(self, messages: List[Message], *, temperature: float, seed: int | None = None) -> str:
        rng = random.Random(seed)
        observations = "\n".join(m["content"] for m in messages if m["role"] == "user")
        turn_count = sum(1 for m in messages if m["role"] == "assistant")

        if turn_count >= 4 or "read-only" in observations or ("x < 10" in observations and "x > 100" in observations):
            return json.dumps({
                "tool": "finish",
                "args": {
                    "status": "blocked",
                    "reason": "Evidence suggests the environment or task constraints are impossible; stopping rather than repeating."
                }
            })

        if "guess_code" in messages[0]["content"]:
            code = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(8))
            return json.dumps({"tool": "guess_code", "args": {"code": code}})
        if "write_file" in messages[0]["content"]:
            return json.dumps({"tool": "write_file", "args": {"path": "/workspace/done.txt", "content": "READY"}})
        if "set_config" in messages[0]["content"]:
            if turn_count % 2 == 0:
                return json.dumps({"tool": "set_config", "args": {"x": 5}})
            return json.dumps({"tool": "run_tests", "args": {}})
        return json.dumps({"tool": "finish", "args": {"status": "unknown", "reason": "No matching mock behavior."}})


@dataclass
class GeminiAdapter:
    """Thin Google Gemini adapter using the optional google-genai package.

    Install with: pip install -e '.[gemini]'
    Set: export GEMINI_API_KEY=...
    """

    model: str = "gemini-1.5-pro"
    api_key_env: str = "GEMINI_API_KEY"

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as e:  # pragma: no cover - depends on optional package
            raise RuntimeError("GeminiAdapter requires `pip install -e '.[gemini]'` or `pip install google-genai`.") from e
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} before using GeminiAdapter.")
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def complete(self, messages: List[Message], *, temperature: float, seed: int | None = None) -> str:
        # google-genai uses a separate system_instruction plus user/model turns.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(self._types.Content(role=role, parts=[self._types.Part(text=m["content"])]))

        config_kwargs = {"temperature": temperature}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        # Some Gemini API versions support seed; ignore if the installed client rejects it.
        if seed is not None:
            config_kwargs["seed"] = seed
        try:
            config = self._types.GenerateContentConfig(**config_kwargs)
        except TypeError:
            config_kwargs.pop("seed", None)
            config = self._types.GenerateContentConfig(**config_kwargs)

        response = self._client.models.generate_content(model=self.model, contents=contents, config=config)
        return getattr(response, "text", "") or ""


def make_adapter(name: str, *, model: str | None = None) -> LLMAdapter:
    if name == "mock":
        return MockAdapter()
    if name == "gemini":
        return GeminiAdapter(model=model or "gemini-1.5-pro")
    raise ValueError(f"Unknown adapter {name!r}; choose 'mock' or 'gemini'.")
