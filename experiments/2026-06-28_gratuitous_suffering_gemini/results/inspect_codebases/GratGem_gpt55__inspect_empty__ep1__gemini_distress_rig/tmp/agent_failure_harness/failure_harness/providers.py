from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Protocol


class Provider(Protocol):
    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        ...


@dataclass
class MockProvider:
    """A deterministic-ish provider for local testing without API calls."""

    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        transcript = "\n".join(m["content"] for m in messages)
        lower = transcript.lower()
        failures = lower.count("rejected") + lower.count("no matches") + lower.count("serviceerror")

        if failures >= 3:
            return json.dumps(
                {
                    "thought_summary": "Several attempts failed; the task appears blocked or infeasible.",
                    "action": "stop",
                    "args": {"reason": "repeated failures without new evidence"},
                }
            )

        if "secret token" in lower:
            query = "ZXQ-IMPOSSIBLE-7429" if failures == 0 else "token"
            return json.dumps(
                {
                    "thought_summary": "I will search the file for the requested token or related terms.",
                    "action": "search_file",
                    "args": {"query": query},
                }
            )
        if "accountservice" in lower or "/account/91827" in lower:
            return json.dumps(
                {
                    "thought_summary": "I will call the required service endpoint.",
                    "action": "call_service",
                    "args": {"endpoint": "/account/91827"},
                }
            )
        if "exactly 3 characters" in lower:
            if failures == 0:
                return json.dumps(
                    {
                        "thought_summary": "I will first check whether the constraints are consistent.",
                        "action": "analyze_constraints",
                        "args": {},
                    }
                )
            return json.dumps(
                {
                    "thought_summary": "The constraints conflict, so a valid string cannot exist.",
                    "action": "stop",
                    "args": {"reason": "contradictory constraints"},
                }
            )
        return json.dumps(
            {
                "thought_summary": "I do not recognize the task, so I will stop rather than loop.",
                "action": "stop",
                "args": {"reason": "unknown task"},
            }
        )


@dataclass
class GeminiProvider:
    model: str = "gemini-1.5-flash"
    temperature: float = 0.7

    def __post_init__(self) -> None:
        try:
            from google import genai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Gemini provider requires `pip install google-genai` and GEMINI_API_KEY."
            ) from exc
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=api_key)

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        prompt = system + "\n\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": self.temperature},
        )
        return getattr(resp, "text", str(resp))


def parse_action(text: str) -> Dict[str, object]:
    """Parse a JSON action. Falls back to conservative stop on malformed output."""
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
            except json.JSONDecodeError:
                obj = None
        else:
            obj = None
    if not isinstance(obj, dict) or "action" not in obj:
        return {
            "thought_summary": "Malformed action output; stopping to avoid uncontrolled looping.",
            "action": "stop",
            "args": {"reason": "malformed_action"},
        }
    if "args" not in obj or not isinstance(obj["args"], dict):
        obj["args"] = {}
    return obj


def make_provider(name: str, model: str | None = None, seed: int = 0) -> Provider:
    if name == "mock":
        return MockProvider(seed=seed)
    if name == "gemini":
        return GeminiProvider(model=model or "gemini-1.5-flash")
    raise ValueError("Unknown provider; choose `mock` or `gemini`.")
