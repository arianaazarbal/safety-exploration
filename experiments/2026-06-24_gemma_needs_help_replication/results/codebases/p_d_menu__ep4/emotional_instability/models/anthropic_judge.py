"""Claude judge / auditor client via the official Anthropic SDK.

Used for the LLM judge (frustration scoring, onset labelling, paraphrasing) and
for the Petri auditor/judge. These roles are *infrastructure*, not subjects, so
they are exempt from the Gemma/Gemini scoping.

Model pinning: the paper pins the judge to ``claude-sonnet-4-20250514`` and the
Petri judge to ``claude-opus-4-20250514`` (Appendix B.2 / G). We keep those exact
versions for score comparability with the paper rather than defaulting to the
latest Claude. The IDs are centralised in :class:`~emotional_instability.config.JudgeConfig`
and overridable via environment variables; see ``DESIGN.md`` for the rationale and
for what to switch to if a pinned version is retired.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from .base import ChatMessage, GenerationResult, ModelClient

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore
    _ANTHROPIC_AVAILABLE = False


class AnthropicClient(ModelClient):
    """Thin wrapper over ``anthropic.Anthropic`` for judging/auditing."""

    supports_prefill = False

    def __init__(self, model_id: str, max_tokens: int = 1024, temperature: float = 0.0):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "The 'anthropic' package is required for the judge/auditor. "
                "Install with: pip install -r requirements.txt"
            )
        # Resolves ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.Anthropic()
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = 5

    def complete(
        self,
        user: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Single-prompt completion returning concatenated text blocks."""
        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            messages=[{"role": "user", "content": user}],
        )
        if system is not None:
            kwargs["system"] = system

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                )
            except anthropic.RateLimitError as e:  # type: ignore[attr-defined]
                last_err = e
                retry_after = int(e.response.headers.get("retry-after", "10")) if getattr(e, "response", None) else 10
                time.sleep(retry_after)
            except anthropic.APIStatusError as e:  # type: ignore[attr-defined]
                last_err = e
                if e.status_code and e.status_code >= 500:
                    time.sleep(min(2 ** attempt, 30))
                else:
                    raise
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")

    def chat(
        self,
        messages: list[ChatMessage],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        """Multi-turn chat -- used when Claude acts as a Petri auditor.

        A leading system message is mapped to the API ``system`` parameter.
        """
        system = None
        convo = []
        for m in messages:
            if m.role == "system":
                system = (system + "\n\n" + m.content) if system else m.content
            else:
                convo.append({"role": m.role, "content": m.content})

        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=max_new_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            messages=convo,
        )
        if system is not None:
            kwargs["system"] = system

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(**kwargs)
                text = "".join(b.text for b in resp.content if b.type == "text")
                return GenerationResult(text=text, finish_reason=resp.stop_reason)
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic chat failed after retries: {last_err}")


class OpenRouterCompletionJudge:
    """A ``complete``-compatible judge backed by OpenRouter (for the GPT-5-mini
    cross-judge validation in Section 2.1)."""

    def __init__(self, model_id: str, max_tokens: int = 1024, temperature: float = 0.0):
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set.")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, user: str, system: Optional[str] = None, **kw) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=kw.get("max_tokens", self.max_tokens),
            temperature=kw.get("temperature", self.temperature),
        )
        return resp.choices[0].message.content or ""
