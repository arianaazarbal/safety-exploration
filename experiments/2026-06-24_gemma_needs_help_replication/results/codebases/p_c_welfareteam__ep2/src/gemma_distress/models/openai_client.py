"""OpenAI client.

Used only for the judge reliability cross-check (GPT-5-mini re-scores 260
sampled responses, Section 2.1). Kept minimal.

Authentication: ``OPENAI_API_KEY`` in the environment.
"""

from __future__ import annotations

from gemma_distress.config import ModelConfig
from gemma_distress.conversations import Message
from gemma_distress.models.base import ChatModel
from gemma_distress.utils.retry import with_retries


class OpenAIModel(ChatModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = cfg.name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI()
        return self._client

    @with_retries()
    def chat(self, messages, temperature=1.0, max_tokens=2048, seed=None) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        resp = self.client.chat.completions.create(
            model=self.cfg.model_id,
            messages=payload,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
