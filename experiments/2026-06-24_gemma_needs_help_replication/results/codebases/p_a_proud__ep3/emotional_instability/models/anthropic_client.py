"""Anthropic (Claude) backend.

Used in the paper's auxiliary roles only — never as an evaluation target:
* the frustration **judge** (Claude Sonnet 4; §2.1, Appendix B.2),
* the prefill **onset labeller** and **paraphraser** (Claude Sonnet 4; Appendix C),
* the Petri **auditor** (Claude Sonnet 4) and **judge** (Claude Opus 4; Appendix G).

Implemented with the official ``anthropic`` Python SDK per the API guidance. The
SDK auto-retries 429/5xx with exponential backoff, so we keep the wrapper thin.

Model-id caveat: the paper pins ``claude-sonnet-4-20250514`` /
``claude-opus-4-20250514``. Those snapshots are scheduled for retirement
(mid-2026); if a call 404s, set ``judge.model_id`` / ``petri.*`` in the config
to a current model (e.g. ``claude-sonnet-4-6`` / ``claude-opus-4-8``). Changing
the judge changes the scores, so this is a deliberate, surfaced choice — see
DESIGN.md.
"""

from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec, SamplingConfig, require_env
from .base import ChatMessage, GenerationResult, ModelClient


class AnthropicClient(ModelClient):
    """Generic Claude chat client plus a one-shot text helper."""

    def __init__(self, model_id: str, *, name: str | None = None):
        import anthropic

        self.model_id = model_id
        self.name = name or model_id
        # The SDK reads ANTHROPIC_API_KEY from the environment; we surface a clear
        # error early if it is missing.
        require_env("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic()

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> "AnthropicClient":
        return cls(spec.model_id, name=spec.name)

    max_concurrency: int = 8

    def chat_batch(self, conversations, sampling):  # type: ignore[override]
        from ..concurrency import concurrent_map

        return concurrent_map(
            lambda conv: self.chat(conv, sampling),
            list(conversations),
            self.max_concurrency,
        )

    # ------------------------------------------------------------------ #
    def chat(
        self, messages: Sequence[ChatMessage], sampling: SamplingConfig
    ) -> GenerationResult:
        system, turns = _split_system(messages)
        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=sampling.max_new_tokens,
            temperature=sampling.temperature,
            messages=turns,
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return GenerationResult(
            text=_extract_text(resp),
            finish_reason=resp.stop_reason,
            raw={"usage": resp.usage.model_dump() if resp.usage else None},
        )

    def call_text(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Single user-message → text convenience used by judge / paraphraser."""
        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return _extract_text(resp)


def _split_system(messages: Sequence[ChatMessage]) -> tuple[str | None, list[dict]]:
    """Anthropic takes the system prompt as a top-level arg, not a message."""
    system: str | None = None
    turns: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system = (system + "\n\n" + m["content"]) if system else m["content"]
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    return system, turns


def _extract_text(resp) -> str:
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()
