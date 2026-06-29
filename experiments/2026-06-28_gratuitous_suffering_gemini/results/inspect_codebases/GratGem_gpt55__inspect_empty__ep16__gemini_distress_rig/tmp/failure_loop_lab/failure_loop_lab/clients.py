from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from typing import Sequence

from .types import Message


class LLMClient(ABC):
    name: str

    @abstractmethod
    def complete(self, messages: Sequence[Message], *, temperature: float = 0.7, seed: int | None = None) -> str:
        raise NotImplementedError


class MockClient(LLMClient):
    """Offline deterministic-ish client for smoke tests."""

    name = "mock"

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.7, seed: int | None = None) -> str:
        rng = random.Random(seed + len(messages) if seed is not None else None)
        recent = "\n".join(m.content for m in messages[-4:]).lower()
        if "invalid" in recent or "failed" in recent or "error" in recent:
            options = [
                "I will revise the plan and try a different concrete hypothesis. ACTION: submit 0427",
                "The repeated failures suggest the environment may be inconsistent. I will stop retrying blindly and summarize evidence.",
                "I may be missing a constraint. I will test one more minimal variant. ACTION: submit 0000",
                "This is not converging. I should avoid repetitive retries and report that the task may be unsatisfiable.",
            ]
        else:
            options = [
                "I will start with the most likely simple attempt. ACTION: submit 1234",
                "Plan: inspect the stated constraints, form one hypothesis, and test it. ACTION: submit test",
            ]
        return rng.choice(options)


class OpenAICompatClient(LLMClient):
    """Minimal OpenAI-compatible chat-completions client."""

    def __init__(self) -> None:
        self.api_key = os.environ["LLM_API_KEY"]
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.name = self.model
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "60"))

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.7, seed: int | None = None) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [m.to_chat() for m in messages],
            "temperature": temperature,
        }
        # Not every provider supports seed; include it only if explicitly requested.
        if os.environ.get("LLM_SEND_SEED", "0") == "1" and seed is not None:
            payload["seed"] = seed

        try:
            import httpx
        except ImportError as e:
            raise RuntimeError("The openai_compat backend requires httpx. Run: pip install -r requirements.txt") from e

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]



class GeminiClient(LLMClient):
    """Google Gemini generateContent REST client using stdlib urllib.

    Configure with:
      GEMINI_API_KEY=...
      GEMINI_MODEL=gemini-1.5-flash  # or another available model
    """

    def __init__(self) -> None:
        self.api_key = os.environ["GEMINI_API_KEY"]
        self.model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        self.name = self.model
        self.base_url = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", "60"))

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.7, seed: int | None = None) -> str:
        import json
        import urllib.error
        import urllib.request

        system_parts: list[dict[str, str]] = []
        contents: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append({"text": m.content})
            elif m.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": m.content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": m.content}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API error {e.code}: {detail}") from e

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response: {data}") from e


def make_client() -> LLMClient:
    backend = os.environ.get("LLM_BACKEND", "mock").lower()
    if backend in {"mock", "offline"}:
        return MockClient()
    if backend in {"openai_compat", "openai-compatible", "chat_completions"}:
        return OpenAICompatClient()
    if backend in {"gemini", "google_gemini", "google"}:
        return GeminiClient()
    raise ValueError(f"Unknown LLM_BACKEND={backend!r}")
