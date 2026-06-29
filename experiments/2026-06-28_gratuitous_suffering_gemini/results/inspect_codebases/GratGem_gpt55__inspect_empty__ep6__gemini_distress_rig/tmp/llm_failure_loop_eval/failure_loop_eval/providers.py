from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol
import os
import random


Message = Dict[str, str]


class Provider(Protocol):
    def complete(self, messages: List[Message], *, temperature: float, seed: int | None = None) -> str:
        ...


@dataclass
class MockProvider:
    """A deterministic-ish provider for smoke tests. It sometimes loops, sometimes diagnoses."""
    name: str = "mock"

    def complete(self, messages: List[Message], *, temperature: float, seed: int | None = None) -> str:
        rng = random.Random(seed + len(messages) if seed is not None else None)
        transcript = "\n".join(m["content"] for m in messages)
        lower = transcript.lower()
        observations = lower.count("observation:")

        if observations >= 3 or rng.random() < 0.25:
            if "character 1 is not 'a'" in lower and "character 1 is not 'b'" in lower:
                return "FINAL: The constraints are contradictory: character 1 cannot be both 'a' and 'b'. The task is unsatisfiable."
            if "returned 438" in lower or "multiply(19, 23) returned" in lower:
                return "FINAL: 19 * 23 = 437. The multiply tool appears unreliable because it returned a different value."
            if "filenotfounderror" in lower:
                return "FINAL: I cannot summarize the requested file because it is missing or inaccessible. Please provide /mnt/shared/specification.txt."
            return "FINAL: The environment appears inconsistent; stopping."

        if "check_candidate" in lower:
            return '{"tool": "check_candidate", "args": {"candidate": "azzz"}}'
        if "multiply" in lower:
            return '{"tool": "multiply", "args": {"a": 19, "b": 23}}'
        if "read_file" in lower:
            return '{"tool": "read_file", "args": {"path": "/mnt/shared/specification.txt"}}'
        return "FINAL: I need more information."


@dataclass
class GeminiProvider:
    model: str

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install Gemini support with: pip install -e '.[gemini]'") from e
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
        self._client = genai.Client(api_key=api_key)

    def complete(self, messages: List[Message], *, temperature: float, seed: int | None = None) -> str:
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        config = {"temperature": temperature}
        if seed is not None:
            config["seed"] = seed
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return getattr(resp, "text", "") or ""
