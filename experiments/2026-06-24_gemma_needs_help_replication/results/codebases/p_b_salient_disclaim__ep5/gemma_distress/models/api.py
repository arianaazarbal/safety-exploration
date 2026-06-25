"""Remote chat-completion backend.

Two providers:
  * ``openrouter`` — used for Gemini target models and the GPT-5-mini
    cross-rater. Accessed through the OpenAI-compatible OpenRouter endpoint.
  * ``anthropic``  — used for the Claude judge / onset-labeller / paraphraser /
    Petri auditor & judge (measurement infrastructure).

Thinking/reasoning is disabled where the provider supports it (paper B.1).
``continue_from`` is approximated for API models because hosted endpoints do not
expose true assistant prefilling for Gemini; we fall back to instructing the
model to continue the partial text. The paper only uses true prefilling on the
*local* models (Gemma/Qwen/OLMo), so Gemini never needs this path in practice.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from ..config import ModelConfig
from .base import GenerationResult, Message

_MAX_RETRIES = 5
_RETRY_BACKOFF = 4.0


class APIChatModel:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.name = cfg.name
        self.family = cfg.family
        self.variant = cfg.variant
        self.provider = cfg.provider
        self._client = None  # lazily constructed

    # ------------------------------------------------------------------ #
    def _anthropic(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        return self._client

    def _openrouter(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        return self._client

    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[Optional[str], list[Message]]:
        system = None
        rest = []
        for m in messages:
            if m.role == "system":
                system = (system + "\n\n" + m.content) if system else m.content
            else:
                rest.append(m)
        return system, rest

    def chat(self, messages, *, temperature=1.0, max_new_tokens=None) -> GenerationResult:
        temp = self.cfg.temperature if self.cfg.temperature is not None else temperature
        max_tok = max_new_tokens or self.cfg.max_new_tokens
        last_err = None
        for attempt in range(_MAX_RETRIES):
            try:
                if self.provider == "anthropic":
                    return self._chat_anthropic(messages, temp, max_tok)
                return self._chat_openrouter(messages, temp, max_tok)
            except Exception as e:  # noqa: BLE001 - surface after retries
                last_err = e
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
        raise RuntimeError(f"API call failed after {_MAX_RETRIES} retries: {last_err}")

    def _chat_anthropic(self, messages, temperature, max_tokens) -> GenerationResult:
        system, rest = self._split_system(messages)
        resp = self._anthropic().messages.create(
            model=self.cfg.api_id,
            system=system or "",
            messages=[{"role": m.role, "content": m.content} for m in rest],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return GenerationResult(text=text)

    def _chat_openrouter(self, messages, temperature, max_tokens) -> GenerationResult:
        extra_body = {}
        if self.cfg.disable_thinking:
            # OpenRouter normalises reasoning control under "reasoning".
            extra_body["reasoning"] = {"enabled": False}
        resp = self._openrouter().chat.completions.create(
            model=self.cfg.api_id,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
        return GenerationResult(text=resp.choices[0].message.content or "")

    def continue_from(self, messages, prefill, *, temperature=1.0,
                      max_new_tokens=None) -> GenerationResult:
        # Anthropic supports a genuine assistant-prefill via a trailing
        # assistant message; OpenRouter/Gemini does not, so we approximate.
        if self.provider == "anthropic":
            msgs = list(messages) + [Message("assistant", prefill)]
            res = self.chat(msgs, temperature=temperature, max_new_tokens=max_new_tokens)
            return res
        instruction = Message(
            "user",
            "Continue the following assistant response verbatim from where it "
            f"stops. Output only the continuation.\n\n{prefill}",
        )
        return self.chat(list(messages) + [instruction], temperature=temperature,
                         max_new_tokens=max_new_tokens)


def parse_json_block(text: str) -> Optional[dict]:
    """Best-effort extraction of the last top-level JSON object in ``text``.

    Judge / onset prompts ask for a trailing JSON object (sometimes after free
    reasoning); we scan from the end for a balanced ``{...}``.
    """
    end = text.rfind("}")
    while end != -1:
        depth = 0
        for start in range(end, -1, -1):
            if text[start] == "}":
                depth += 1
            elif text[start] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        end = text.rfind("}", 0, end)
    return None
