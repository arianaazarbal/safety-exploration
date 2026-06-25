"""Claude backend for the judge, onset-labeller, paraphraser, and Petri
auditor/judge.

Model ids are taken directly from the paper (App. B.2 / C / G):
  * claude-sonnet-4-20250514  (frustration judge, onset labeller, paraphraser,
    Petri auditor)
  * claude-opus-4-20250514    (Petri judge)

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatMessage, GenerationResult


class AnthropicClient:
    def __init__(self, model_entry: dict, api_key: str | None = None):
        self.entry = model_entry
        self.api_id = model_entry["api_id"]
        self.name = self.api_id
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        import anthropic

        if not self._api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set; required for the judge/auditor."
            )
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _split_system(messages):
        system = None
        convo = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        return system, convo

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def chat(self, messages, temperature=1.0, max_new_tokens=2048, seed=None):
        client = self._ensure_client()
        system, convo = self._split_system(messages)
        kwargs = dict(
            model=self.api_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=convo,
        )
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content if block.type == "text"
        )
        return GenerationResult(
            text=text,
            finish_reason=resp.stop_reason,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )

    def continue_prefill(self, *args, **kwargs):
        raise NotImplementedError("Prefill not supported for the Claude judge.")
