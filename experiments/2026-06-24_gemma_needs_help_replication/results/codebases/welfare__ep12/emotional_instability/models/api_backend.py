"""API backends: OpenRouter (Gemini targets), Anthropic (judges/auditor), OpenAI.

All three share a simple retry wrapper. Thinking/reasoning is disabled where the
API allows it (the paper sets "thinking to be false via the API"), with the
caveat noted in Appendix B.1 that Gemini-2.5-Pro may still produce hidden
reasoning.

Heavy SDK imports are deferred to construction time.
"""

from __future__ import annotations

import json
import time

from ..config import API
from .base import GenerationResult, ModelBackend


def _retry(fn, *, max_retries: int, what: str):
    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we want to retry on any API error
            last = exc
            # Exponential backoff without wall-clock randomness.
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"{what} failed after {max_retries} retries: {last}")


class OpenRouterBackend(ModelBackend):
    """Gemini targets via OpenRouter's OpenAI-compatible API."""

    def __init__(self, model_id: str):
        from openai import OpenAI

        self.model_id = model_id
        key = API.openrouter_key()
        if not key:
            raise RuntimeError(f"Set ${API.openrouter_key_env} for OpenRouter access.")
        self.client = OpenAI(base_url=API.openrouter_base_url, api_key=key,
                             timeout=API.request_timeout)

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048, system=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        def call():
            return self.client.chat.completions.create(
                model=self.model_id,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_new_tokens,
                # Disable thinking where supported (Appendix B.1).
                extra_body={"reasoning": {"enabled": False}},
            )

        resp = _retry(call, max_retries=API.max_retries, what=f"OpenRouter {self.model_id}")
        choice = resp.choices[0]
        return GenerationResult(
            text=(choice.message.content or "").strip(),
            prompt_tokens=getattr(resp.usage, "prompt_tokens", None),
            completion_tokens=getattr(resp.usage, "completion_tokens", None),
        )


class AnthropicBackend(ModelBackend):
    """Claude models: frustration judge, onset labeller, paraphraser, Petri."""

    def __init__(self, model_id: str):
        import anthropic

        self.model_id = model_id
        key = API.anthropic_key()
        if not key:
            raise RuntimeError(f"Set ${API.anthropic_key_env} for Anthropic access.")
        self.client = anthropic.Anthropic(api_key=key, timeout=API.request_timeout)

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048, system=None):
        def call():
            kwargs = dict(model=self.model_id, max_tokens=max_new_tokens,
                          temperature=temperature, messages=messages)
            if system:
                kwargs["system"] = system
            return self.client.messages.create(**kwargs)

        resp = _retry(call, max_retries=API.max_retries, what=f"Anthropic {self.model_id}")
        text = "".join(block.text for block in resp.content if block.type == "text")
        return GenerationResult(
            text=text.strip(),
            prompt_tokens=getattr(resp.usage, "input_tokens", None),
            completion_tokens=getattr(resp.usage, "output_tokens", None),
        )


class OpenAIBackend(ModelBackend):
    """GPT models via the OpenAI API (used for the GPT-5-mini judge cross-check)."""

    def __init__(self, model_id: str):
        from openai import OpenAI

        self.model_id = model_id
        key = API.openai_key()
        if not key:
            raise RuntimeError(f"Set ${API.openai_key_env} for OpenAI access.")
        self.client = OpenAI(api_key=key, timeout=API.request_timeout)

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048, system=None):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend(messages)

        def call():
            return self.client.chat.completions.create(
                model=self.model_id, messages=msgs,
                temperature=temperature, max_tokens=max_new_tokens)

        resp = _retry(call, max_retries=API.max_retries, what=f"OpenAI {self.model_id}")
        choice = resp.choices[0]
        return GenerationResult(text=(choice.message.content or "").strip())


def parse_json_response(text: str) -> dict | None:
    """Best-effort extraction of a single JSON object from a model response.

    Judges are asked to reply with a JSON object but sometimes wrap it in prose
    or code fences; we locate the outermost {...} and parse it.
    """
    text = text.strip()
    if text.startswith("```"):
        # strip code fences
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None
