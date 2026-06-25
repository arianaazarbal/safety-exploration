"""Unified thin clients for the API providers used as judges/auditors/targets.

Three providers are needed:
  * anthropic  - Claude judge, onset-labeller, paraphraser, Petri auditor & judge.
  * openai     - GPT-5-mini secondary judge.
  * gemini     - Gemini target models. We support BOTH the native google-genai SDK
                 and OpenRouter (the paper used OpenRouter); selected by env/base_url.

Every call goes through tenacity retry with exponential backoff because the
multi-thousand-sample sweeps will inevitably hit transient rate limits.

API keys are read from the environment:
  ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (or GOOGLE_API_KEY),
  OPENROUTER_API_KEY.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_random_exponential

# Roles are normalised to {"role": "user"|"assistant"|"system", "content": str}.
Message = dict[str, str]


@dataclass
class ChatResult:
    text: str
    raw: Any = None


_RETRY = retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)


# --------------------------------------------------------------------------- #
# Anthropic                                                                    #
# --------------------------------------------------------------------------- #
class AnthropicClient:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024):
        import anthropic

        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @_RETRY
    def chat(self, messages: list[Message], system: str | None = None, **kw: Any) -> ChatResult:
        sys_text = system
        conv: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                sys_text = (sys_text + "\n\n" + m["content"]) if sys_text else m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        resp = self._client.messages.create(
            model=self.model,
            system=sys_text or anthropic_NOT_GIVEN(),
            messages=conv,
            temperature=kw.get("temperature", self.temperature),
            max_tokens=kw.get("max_tokens", self.max_tokens),
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ChatResult(text=text, raw=resp)


def anthropic_NOT_GIVEN():  # pragma: no cover - tiny shim to avoid hard import at module top
    import anthropic

    return anthropic.NOT_GIVEN


# --------------------------------------------------------------------------- #
# OpenAI (also serves OpenRouter via base_url)                                 #
# --------------------------------------------------------------------------- #
class OpenAIClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        extra_body: dict | None = None,
    ):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=os.environ.get(api_key_env),
            base_url=base_url,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    @_RETRY
    def chat(self, messages: list[Message], system: str | None = None, **kw: Any) -> ChatResult:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=kw.get("temperature", self.temperature),
            max_tokens=kw.get("max_tokens", self.max_tokens),
            extra_body={**self.extra_body, **kw.get("extra_body", {})} or None,
        )
        return ChatResult(text=resp.choices[0].message.content or "", raw=resp)


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def make_client(provider: str, model: str, **kw: Any):
    provider = provider.lower()
    if provider == "anthropic":
        return AnthropicClient(model=model, **kw)
    if provider == "openai":
        return OpenAIClient(model=model, **kw)
    if provider == "openrouter":
        return OpenAIClient(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            **kw,
        )
    raise ValueError(f"Unknown provider: {provider}")
