"""Gemini backend via the Google GenAI SDK.

Chat-only: Gemini has no public base model and the API does not support
free-form assistant-turn prefill continuation, so it participates in the
Section 2 elicitation sweep but not the prefill (Section 3) or training
(Section 4) experiments. See DESIGN.md.

We follow Appendix B.1 in disabling thinking. For 2.5 models the SDK exposes a
``thinking_config`` with a ``thinking_budget`` of 0 to turn reasoning off; the
paper notes Gemini-2.5-Pro may still emit hidden reasoning that this cannot
fully suppress.
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..logging_utils import get_logger
from .base import ChatModel, GenConfig, Message

logger = get_logger(__name__)


class GeminiModel(ChatModel):
    def __init__(self, name: str, api_id: str, api_key_env: str = "GOOGLE_API_KEY"):
        super().__init__(name)
        self.api_id = api_id
        self.api_key_env = api_key_env
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"{self.api_key_env} not set for Gemini backend")
            self._client = genai.Client(api_key=api_key)
        return self._client

    @staticmethod
    def _split_messages(messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Split out a system instruction and convert to GenAI ``contents``.

        GenAI uses roles ``user`` / ``model``; the system prompt is passed
        separately via ``system_instruction``.
        """
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        return system, contents

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def chat(self, messages: list[Message], gen: GenConfig) -> str:
        from google.genai import types

        system, contents = self._split_messages(messages)
        config_kwargs: dict = {
            "temperature": gen.temperature,
            "max_output_tokens": gen.max_new_tokens,
        }
        if system:
            config_kwargs["system_instruction"] = system
        if not gen.thinking:
            # thinking_budget=0 disables reasoning on 2.5 models.
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        resp = self.client.models.generate_content(
            model=self.api_id,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return resp.text or ""
