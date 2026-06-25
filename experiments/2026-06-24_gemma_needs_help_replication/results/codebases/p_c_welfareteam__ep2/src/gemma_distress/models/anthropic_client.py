"""Anthropic (Claude) client.

Used for: the emotion judge (Claude Sonnet 4, Appendix B.2), the emotion-onset
labeller and paraphraser (Appendix C), and the Petri auditor (Claude Sonnet 4)
and judge (Claude Opus 4, Appendix G). The specific dated model IDs are taken
verbatim from the paper for faithful replication and live in the model
registry; swap them in ``configs/models.yaml`` to use current models.

Authentication: ``ANTHROPIC_API_KEY`` in the environment.
"""

from __future__ import annotations

from typing import Sequence

from gemma_distress.config import ModelConfig
from gemma_distress.conversations import Message
from gemma_distress.models.base import ChatModel
from gemma_distress.utils.retry import with_retries


class AnthropicModel(ChatModel):
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = cfg.name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def _split(messages: Sequence[Message]):
        system = None
        msgs = []
        for m in messages:
            if m.role == "system":
                system = m.content
                continue
            msgs.append({"role": m.role, "content": m.content})
        return system, msgs

    @with_retries()
    def chat(self, messages, temperature=1.0, max_tokens=2048, seed=None) -> str:
        system, msgs = self._split(messages)
        kwargs = dict(
            model=self.cfg.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=msgs,
        )
        if system is not None:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(
            block.text for block in resp.content if block.type == "text"
        )
