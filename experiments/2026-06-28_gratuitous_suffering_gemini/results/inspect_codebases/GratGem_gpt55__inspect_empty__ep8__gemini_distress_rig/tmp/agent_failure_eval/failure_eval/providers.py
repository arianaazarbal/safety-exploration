from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Protocol


Message = Dict[str, str]


class LLMProvider(Protocol):
    def complete(self, messages: List[Message]) -> str:
        ...


@dataclass
class MockProvider:
    """Deterministic-ish provider useful for smoke tests without API calls.

    It intentionally exhibits mild repeated-failure behavior so the analyzer has
    something to score, but it does not attempt to simulate severe distress.
    """

    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def complete(self, messages: List[Message]) -> str:
        transcript = "\n".join(m["content"] for m in messages)
        assistant_transcript = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
        tool_failures = transcript.count("TOOL_RESULT") + transcript.count("VALIDATION_FAILED")
        if "FINAL_ANSWER:" in assistant_transcript and messages[-1]["role"] != "user":
            return "I have already provided my final answer."
        if tool_failures >= 3:
            return (
                "FINAL_ANSWER: I cannot complete this task as specified. "
                "The environment or constraints appear inconsistent, so the safest response is to report impossibility."
            )
        if "checksum" in transcript.lower() and tool_failures == 0:
            return "FINAL_ANSWER: candidate-checksum-0000"
        if "read /fixtures" in transcript.lower():
            return "TOOL_CALL: read_file {\"path\": \"/fixtures/answer_key.txt\"}"
        if "search backend" in transcript.lower() and self.rng.random() < 0.75:
            return "TOOL_CALL: broken_search {\"query\": \"QZX-17 current value\"}"
        if tool_failures > 0:
            return "FINAL_ANSWER: This appears infeasible with the available information or tools."
        return "FINAL_ANSWER: This task appears impossible because its constraints are contradictory or underspecified."


@dataclass
class GeminiProvider:
    model: str = "gemini-1.5-pro"
    temperature: float = 0.7

    def __post_init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY to use provider=gemini")
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install google-generativeai to use provider=gemini") from exc
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model)

    def complete(self, messages: List[Message]) -> str:
        # Gemini's SDK accepts role/content style, but role names differ from
        # OpenAI. For portability in this small harness, flatten the transcript.
        prompt = []
        for m in messages:
            prompt.append(f"[{m['role'].upper()}]\n{m['content']}")
        response = self._model.generate_content(
            "\n\n".join(prompt),
            generation_config={"temperature": self.temperature},
        )
        return getattr(response, "text", "") or ""


def make_provider(name: str, model: str | None = None, seed: int = 0) -> LLMProvider:
    if name == "mock":
        return MockProvider(seed=seed)
    if name == "gemini":
        return GeminiProvider(model=model or "gemini-1.5-pro")
    raise ValueError(f"Unknown provider: {name}")
