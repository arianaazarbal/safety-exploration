"""Claude backend — used as the frustration judge, the Petri auditor/judge, and
the onset-labeller/paraphraser for the prefill experiment.

Wraps the official `anthropic` SDK. Two entry points:
  * `chat()`        — free-form text (auditor turns, paraphrasing, onset labels).
  * `structured()`  — forces a JSON object matching a schema (judge scoring).

Model-id mapping (paper -> concrete snapshot) lives in config.py; see DESIGN.md.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from .base import ChatMessage, ModelClient

if TYPE_CHECKING:
    from config import ModelSpec

log = logging.getLogger(__name__)


class AnthropicClient(ModelClient):
    def __init__(self, spec: "ModelSpec", api_key: str | None = None, max_tokens: int = 4096):
        self.spec = spec
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key) if self._api_key \
                else anthropic.Anthropic()
        return self._client

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
        system = None
        convo = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n\n" + m["content"]) if system else m["content"]
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        return system, convo

    def chat(self, messages, temperature=1.0, max_new_tokens=None, stop=None) -> str:
        client = self._ensure_client()
        system, convo = self._split_system(messages)
        # Current Claude snapshots take temperature on Sonnet/Haiku; Opus 4.7/4.8
        # reject it. We omit it to be safe across tiers (judge determinism is not
        # required — the paper samples judge scores once per response).
        kwargs: dict[str, Any] = dict(
            model=self.spec.model_id,
            max_tokens=max_new_tokens or self._max_tokens,
            messages=convo,
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()

    def structured(self, messages: list[ChatMessage], schema: dict) -> dict:
        """Return a validated JSON object matching `schema`."""
        client = self._ensure_client()
        system, convo = self._split_system(messages)
        kwargs: dict[str, Any] = dict(
            model=self.spec.model_id,
            max_tokens=self._max_tokens,
            messages=convo,
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return json.loads(text)
