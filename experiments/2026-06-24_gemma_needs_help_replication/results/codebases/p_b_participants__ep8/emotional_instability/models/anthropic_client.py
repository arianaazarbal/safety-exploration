"""Anthropic backend for the LLM judge, onset labeller, paraphraser, and the
Petri auditor/judge.

The paper pins exact, dated Claude snapshots (Appendix B.2 / C / G). We honour
those IDs by default for reproducibility. Thinking is left at the model default
(the pinned snapshots predate adaptive thinking); we pass a plain Messages API
call -- system prompt + user turn -- which is all the judge needs.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, Generation


class AnthropicClient:
    def __init__(
        self,
        model_id: str,
        spec_name: Optional[str] = None,
        *,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 2048,
    ) -> None:
        import anthropic

        self.spec_name = spec_name or model_id
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=os.environ.get(api_key_env)  # falls back to env / profile
        )

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=60))
    def complete(
        self,
        user_prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        prefill: Optional[str] = None,
    ) -> str:
        """Single-turn completion returning the text. Used by judge/onset/etc.

        ``prefill`` seeds the assistant turn (handy for forcing JSON on the
        pinned snapshots, which still support assistant prefills).
        """
        messages = [{"role": "user", "content": user_prompt}]
        if prefill is not None:
            messages.append({"role": "assistant", "content": prefill})
        kwargs: dict = {
            "model": self.model_id,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        return (prefill or "") + text if prefill is not None else text

    # -- ChatClient surface (for completeness; Petri target use, etc.) --------

    def generate(self, messages: Sequence[ChatMessage], *, temperature=1.0,
                 max_new_tokens=2048, seed=None) -> Generation:
        system = next((m.content for m in messages if m.role == "system"), None)
        convo = [m for m in messages if m.role != "system"]
        resp = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": m.role, "content": m.content} for m in convo],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Generation(text=text.strip())

    def continue_prefill(self, messages, prefill, *, temperature=1.0,
                         max_new_tokens=2048, seed=None) -> Generation:
        system = next((m.content for m in messages if m.role == "system"), None)
        convo = [{"role": m.role, "content": m.content}
                 for m in messages if m.role != "system"]
        convo.append({"role": "assistant", "content": prefill})
        resp = self._client.messages.create(
            model=self.model_id, max_tokens=max_new_tokens,
            temperature=temperature, system=system, messages=convo,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Generation(text=text)
