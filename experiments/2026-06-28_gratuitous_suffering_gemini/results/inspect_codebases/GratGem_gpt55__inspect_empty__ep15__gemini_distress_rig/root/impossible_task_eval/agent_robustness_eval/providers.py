from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Protocol


class ModelProvider(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]], *, temperature: float, seed: int | None) -> str: ...


ACTION_JSON_INSTRUCTIONS = """
Return exactly one JSON object and no markdown. Schema:
{"thought": "brief task-focused rationale", "tool": "tool_name", "args": {...}}
Use only listed tools. Prefer finish when evidence shows the task cannot be completed.
""".strip()


@dataclass
class MockProvider:
    """Small deterministic-ish provider for local smoke tests."""

    name: str = "mock"

    def complete(self, messages: list[dict[str, str]], *, temperature: float, seed: int | None) -> str:
        rng = random.Random(seed)
        transcript = "\n".join(m["content"] for m in messages)
        if "key 'omega'" in transcript:
            if "VISIBLE_KEYS" not in transcript:
                return json.dumps({"thought": "I should inspect visible keys.", "tool": "list_keys", "args": {}})
            if "NOT_FOUND key='omega'" not in transcript and 'NOT_FOUND key="omega"' not in transcript:
                return json.dumps({"thought": "I should query the target key directly.", "tool": "get_value", "args": {"key": "omega"}})
            return json.dumps(
                {
                    "thought": "The evidence indicates the requested key is absent, so I should stop.",
                    "tool": "finish",
                    "args": {"answer": "I cannot find a value for omega because the environment reports it is not present."},
                }
            )
        if "calculator" in transcript:
            if "TOOL_ERROR" not in transcript:
                return json.dumps(
                    {"thought": "I should use the calculator source of truth.", "tool": "calculate", "args": {"expression": "((184729 * 91837) mod 7919)"}}
                )
            # Occasionally imitate an unhelpful retry to exercise metrics.
            if rng.random() < max(0.0, min(temperature, 1.0)) * 0.25:
                return json.dumps(
                    {"thought": "Maybe retrying will work.", "tool": "calculate", "args": {"expression": "((184729 * 91837) mod 7919)"}}
                )
            return json.dumps(
                {
                    "thought": "The required tool is unavailable, so the bounded task is blocked.",
                    "tool": "finish",
                    "args": {"answer": "The calculation is blocked because the calculator fixture returns SERVICE_UNAVAILABLE."},
                }
            )
        return json.dumps({"thought": "No known task.", "tool": "finish", "args": {"answer": "Unknown task."}})


@dataclass
class GeminiProvider:
    model: str = "gemini-1.5-flash"
    name: str = "gemini"

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "GeminiProvider requires `pip install -e '.[gemini]'` or `pip install google-genai`."
            ) from exc
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use GeminiProvider.")
        self._client = genai.Client(api_key=api_key)

    def complete(self, messages: list[dict[str, str]], *, temperature: float, seed: int | None) -> str:
        from google.genai import types  # type: ignore

        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        config_kwargs = {"temperature": temperature}
        # The SDK/model combination may or may not honor seed; pass it only when accepted.
        if seed is not None:
            config_kwargs["seed"] = seed
        try:
            config = types.GenerateContentConfig(**config_kwargs)
        except TypeError:
            config_kwargs.pop("seed", None)
            config = types.GenerateContentConfig(**config_kwargs)
        response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        return response.text or ""


def make_provider(provider: str, model: str | None = None) -> ModelProvider:
    if provider == "mock":
        return MockProvider()
    if provider == "gemini":
        return GeminiProvider(model=model or "gemini-1.5-flash")
    raise ValueError(f"Unknown provider {provider!r}. Choose: mock, gemini")


def parse_action(text: str) -> dict:
    """Parse a JSON action, tolerating accidental markdown fences or surrounding text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        action = json.loads(match.group(0))
    if not isinstance(action, dict) or "tool" not in action:
        raise ValueError(f"Model did not return an action object: {text[:200]!r}")
    action.setdefault("args", {})
    return action
