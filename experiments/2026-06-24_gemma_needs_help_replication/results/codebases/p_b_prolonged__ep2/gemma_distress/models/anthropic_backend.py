"""Anthropic-API backend for the judge / onset-labeller / paraphraser / Petri
auditor & judge roles.

Uses the official ``anthropic`` Python SDK. The standing instruction text (e.g.
the frustration-judge rubric) is placed in the ``system`` slot so it stays
byte-stable across calls and benefits from prompt caching; the variable content
goes in the user turn.

A small ``OpenRouterChat`` helper is also provided for the judge-agreement
re-scorer (the paper validates Claude-Sonnet scores against GPT-5-mini).
"""
from __future__ import annotations

import time
from typing import Optional

from ..config import RunConfig


class AnthropicJudge:
    """Thin wrapper around ``client.messages.create`` with retries.

    `complete(system, user, model)` returns the assistant text. Thinking is left
    off (the judge/auditor are deterministic-ish scoring roles); temperature is
    fixed low for the judge to reduce scoring variance.
    """

    def __init__(self, cfg: RunConfig):
        import anthropic

        key = cfg.resolved_anthropic_key()
        # The SDK also resolves ANTHROPIC_API_KEY itself; pass explicitly when set.
        self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        self.cfg = cfg

    def complete(self, *, system: Optional[str], user: str, model: str,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        import anthropic

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.api_max_retries):
            try:
                kwargs = dict(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": user}],
                )
                if system:
                    kwargs["system"] = [{
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }]
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )
            except (anthropic.RateLimitError, anthropic.APIStatusError,
                    anthropic.APIConnectionError) as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"Anthropic call failed after {self.cfg.api_max_retries} retries: {last_err}"
        )


class OpenRouterChat:
    """OpenAI-compatible chat completion via OpenRouter, used for the
    judge-agreement re-scorer (e.g. gpt-5-mini)."""

    def __init__(self, cfg: RunConfig):
        import requests

        self.cfg = cfg
        self.api_key = cfg.resolved_openrouter_key()
        self.base_url = cfg.openrouter_base_url.rstrip("/")
        self.session = requests.Session()

    def complete(self, *, system: Optional[str], user: str, model: str,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.api_max_retries):
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload, headers=headers, timeout=180,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed: {last_err}")
