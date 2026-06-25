"""OpenRouter-backed API model (OpenAI-compatible client).

The paper reaches all API models (Gemini, the Claude judge/auditor, GPT) through
OpenRouter (Appendix B.1), so we do the same for a single uniform code path. The
`google-genai` / `anthropic` native SDKs would also work; OpenRouter keeps one
client and one key.

Thinking is disabled where the model supports it (§B.1: "we set thinking to be
false via the API"). OpenRouter exposes this through the unified `reasoning`
field (`{"enabled": false}` / `{"max_tokens": 0}`); we also pass the Gemini
`thinking_budget` hint through `extra_body` for good measure.
"""
from __future__ import annotations

import time

from .. import config
from .base import LLM, GenConfig, Message


class OpenRouterModel(LLM):
    def __init__(self, name: str, model_id: str, is_instruct: bool = True,
                 disable_thinking: bool = True, max_retries: int = 6):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai to use API models") from e
        self.name = name
        self.model_id = model_id
        self.is_instruct = is_instruct
        self.disable_thinking = disable_thinking
        self.max_retries = max_retries
        self._client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.openrouter_api_key(),
        )

    def _extra_body(self) -> dict:
        if not self.disable_thinking:
            return {}
        # OpenRouter's unified reasoning control (§B.1: "thinking set to false via
        # the API"). `enabled: false` disables reasoning where the provider honours
        # it; `max_tokens: 0` is a belt-and-braces hint. These keys are merged into
        # the top level of the request body by the OpenAI client's `extra_body`.
        # The paper notes Gemini-2.5-Pro / GPT-5.2 may still emit hidden reasoning
        # that no API flag prevents (recorded in the relevant ModelSpec.notes).
        return {"reasoning": {"enabled": False, "max_tokens": 0}}

    def chat(self, messages: list[Message], cfg: GenConfig | None = None) -> str:
        cfg = cfg or GenConfig()
        kwargs: dict = dict(
            model=self.model_id,
            messages=list(messages),
            temperature=cfg.temperature,
            max_tokens=cfg.max_new_tokens,
            top_p=cfg.top_p,
        )
        if cfg.stop:
            kwargs["stop"] = list(cfg.stop)
        if cfg.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if cfg.seed is not None:
            kwargs["seed"] = cfg.seed
        extra = self._extra_body()
        if extra:
            kwargs["extra_body"] = extra

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                return content or ""
            except Exception as e:  # rate limits, transient 5xx, etc.
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(
            f"OpenRouter call failed after {self.max_retries} retries for "
            f"{self.model_id}: {last_err}")
