"""OpenAI-compatible API client (OpenRouter) for Gemini targets and the LLM
judges.

The paper queries closed models through OpenRouter and disables thinking where
possible.  We use the OpenAI SDK pointed at the OpenRouter base URL, which
serves Gemini, Claude and GPT behind one interface.
"""

from __future__ import annotations

import json

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import RUNTIME, ModelSpec
from .base import Message, ModelClient


def _client():
    from openai import OpenAI

    if not RUNTIME.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; required for API-backed models."
        )
    return OpenAI(
        base_url=RUNTIME.openrouter_base_url,
        api_key=RUNTIME.openrouter_api_key,
    )


# Provider-specific knobs to disable hidden reasoning where supported
# (Appendix B.1: "we set thinking to be false via the API").
def _thinking_off_extra(model_id: str) -> dict:
    extra: dict = {}
    if "gemini" in model_id:
        # OpenRouter passes provider-specific fields through `extra_body`.
        extra["extra_body"] = {
            "reasoning": {"enabled": False},
            "google": {"thinking_config": {"thinking_budget": 0}},
        }
    return extra


class APIModelClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec.name)
        self.spec = spec
        self.model_id = spec.model_id
        self._oai = _client()

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call(self, messages, temperature, max_new_tokens) -> str:
        resp = self._oai.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            **_thinking_off_extra(self.model_id),
        )
        return resp.choices[0].message.content or ""

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_new_tokens: int,
        prefill: str | None = None,
    ) -> str:
        msgs = [dict(m) for m in messages]
        if prefill:
            # Assistant-prefill via a trailing assistant message. Not all
            # providers honour this; the prefill study targets local Gemma.
            msgs.append({"role": "assistant", "content": prefill})
        out = self._call(msgs, temperature, max_new_tokens)
        return out


# --------------------------------------------------------------------------
# Lightweight chat helper used by judges / auditors (no ModelSpec needed)
# --------------------------------------------------------------------------
class ChatAPI:
    """Thin wrapper for one-off judge / auditor calls by model id."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._oai = _client()

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    def complete(self, messages, *, temperature: float = 0.0,
                 max_tokens: int = 1024) -> str:
        resp = self._oai.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def complete_json(self, messages, *, temperature: float = 0.0,
                      max_tokens: int = 1024) -> dict:
        """Call and parse the first JSON object in the response."""
        raw = self.complete(messages, temperature=temperature,
                            max_tokens=max_tokens)
        return _extract_json(raw)


def _extract_json(text: str) -> dict:
    """Best-effort extraction of the last balanced JSON object in `text`."""
    # find the last closing brace and walk back to its matching open brace
    end = text.rfind("}")
    if end == -1:
        raise ValueError(f"no JSON object in judge output: {text[:200]!r}")
    depth = 0
    for start in range(end, -1, -1):
        if text[start] == "}":
            depth += 1
        elif text[start] == "{":
            depth -= 1
            if depth == 0:
                candidate = text[start:end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # keep scanning further back for a valid object
                    continue
    raise ValueError(f"could not parse JSON from: {text[:200]!r}")
