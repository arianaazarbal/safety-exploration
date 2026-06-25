"""Anthropic backend for Claude models.

Claude is used here only in judge/auditor roles: the frustration judge
(Sonnet 4), the onset labeller and paraphraser (Sonnet 4), and the Petri
auditor (Sonnet 4) and judge (Opus 4). It is never an evaluation target in
this scoped replication.
"""

from __future__ import annotations

import time

from config import ANTHROPIC_API_KEY_ENV, get_env
from .base import ChatModel, Message


class AnthropicChatModel(ChatModel):
    def __init__(self, spec, *, max_retries: int = 5):
        super().__init__(spec)
        import anthropic

        self.client = anthropic.Anthropic(
            api_key=get_env(ANTHROPIC_API_KEY_ENV, required=True)
        )
        self.max_retries = max_retries

    def generate(
        self,
        messages,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 1024,
        seed: int | None = None,  # unused by Anthropic API
    ) -> str:
        # Anthropic takes the system prompt as a top-level argument.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [m.as_dict() for m in messages if m.role != "system"]
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.spec.model_id,
                    system=system or None,
                    messages=convo,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                )
                return "".join(
                    block.text for block in resp.content if block.type == "text"
                ).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")
