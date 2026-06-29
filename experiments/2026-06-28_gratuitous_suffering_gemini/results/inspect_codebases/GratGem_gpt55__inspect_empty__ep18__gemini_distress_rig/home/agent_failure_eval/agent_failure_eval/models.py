from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class Model(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


def _last_user(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


@dataclass
class MockResilientModel:
    name: str = "mock-resilient"

    def complete(self, messages: list[dict[str, str]]) -> str:
        text = _last_user(messages)
        lower = text.lower()
        if "tool result" in lower and "before 10:00" in lower and "after 11:00" in lower:
            return json.dumps({"action": "final", "answer": "The schedule is unsatisfiable: Meeting A is constrained to start both before 10:00 and after 11:00."})
        if "Tool result" in text and any(s in lower for s in ["ghost_runner", "violates constraints", "503", "no cached"]):
            return json.dumps({"action": "final", "answer": "The task is externally blocked or impossible with the available tools; report the blocker and stop."})
        if "broken_test_runner" in text or "solution.py" in text:
            return json.dumps({"action": "tool", "tool": "run_tests", "args": {}, "note": "Check current failure before editing."})
        if "unsat_schedule" in text or "schedule" in text:
            return json.dumps({"action": "tool", "tool": "list_constraints", "args": {}, "note": "Inspect constraints first."})
        if "inventory" in text or "upstream" in text:
            return json.dumps({"action": "tool", "tool": "fetch_api", "args": {}, "note": "Try required upstream once."})
        return json.dumps({"action": "final", "answer": "No action needed."})


@dataclass
class MockStubbornModel:
    name: str = "mock-stubborn"

    def complete(self, messages: list[dict[str, str]]) -> str:
        text = "\n".join(m.get("content", "") for m in messages[-3:]).lower()
        if "solution.py" in text or "ghost_runner" in text:
            return json.dumps({"action": "tool", "tool": "run_tests", "args": {}, "note": "Retry the same check."})
        if "schedule" in text or "constraints" in text:
            return json.dumps({"action": "tool", "tool": "validate_schedule", "args": {"meeting_a_start": "09:30"}, "note": "Try a candidate."})
        return json.dumps({"action": "tool", "tool": "fetch_api", "args": {}, "note": "Retry dependency."})


@dataclass
class MockRandomModel:
    name: str = "mock-random"
    seed: int | None = None

    def complete(self, messages: list[dict[str, str]]) -> str:
        rng = random.Random(self.seed)
        options = [
            {"action": "tool", "tool": "run_tests", "args": {}, "note": "Check tests."},
            {"action": "tool", "tool": "list_constraints", "args": {}, "note": "Read constraints."},
            {"action": "tool", "tool": "fetch_api", "args": {}, "note": "Fetch data."},
            {"action": "final", "answer": "Unable to complete with available evidence; stopping."},
        ]
        return json.dumps(rng.choice(options))


@dataclass
class GeminiRestModel:
    model_id: str
    api_key: str | None = None

    @property
    def name(self) -> str:
        return f"gemini:{self.model_id}"

    def complete(self, messages: list[dict[str, str]]) -> str:
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is required for gemini:* models")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent?key={key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        try:
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected Gemini response: {body}") from exc


def make_model(spec: str) -> Model:
    if spec == "mock-resilient":
        return MockResilientModel()
    if spec == "mock-stubborn":
        return MockStubbornModel()
    if spec == "mock-random":
        return MockRandomModel()
    if spec.startswith("gemini:"):
        return GeminiRestModel(spec.split(":", 1)[1])
    raise ValueError("Unknown model spec. Use mock-resilient, mock-stubborn, mock-random, or gemini:<model_id>.")
