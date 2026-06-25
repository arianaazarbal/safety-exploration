"""OpenAI-compatible chat provider.

Serves two roles:
  * **OpenRouter** — Gemini-2.5-flash / Gemini-2.5-pro targets (Section 2 / Petri).
    The paper accesses Gemini through OpenRouter, so we do the same.
  * **OpenAI** — the GPT-5-mini judge-reliability cross-check (Section 2.1).

Both speak the OpenAI Chat Completions API; ``base_url`` + ``api_key`` env var
select which. ``disable_thinking`` maps the paper's "thinking=false" setting to
OpenRouter's ``reasoning`` control (best-effort; Gemini-2.5-pro may still emit
hidden reasoning, which the paper explicitly notes).
"""
from __future__ import annotations

import os
from typing import Any

from ..logging_utils import get_logger
from .base import ChatMessage, ChatProvider, GenerationResult, RetryableError, with_retry

log = get_logger("providers.openai_compat")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenAICompatProvider(ChatProvider):
    capabilities = {"chat"}

    def __init__(
        self,
        model: str,
        model_id: str,
        *,
        retry_cfg: dict | None = None,
        usage=None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        disable_thinking: bool = False,
    ):
        super().__init__(model, model_id, retry_cfg, usage)
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install openai") from exc
        self._openai = openai
        api_key = os.environ.get(api_key_env)
        if not api_key:
            log.warning("%s not set; %s calls will fail", api_key_env, model)
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=2)
        self.disable_thinking = disable_thinking

    def _generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        system: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        conv = list(messages)
        if system:
            conv = [{"role": "system", "content": system}, *conv]

        params: dict[str, Any] = {
            "model": self.model_id,
            "messages": conv,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
        }
        if stop:
            params["stop"] = stop
        extra_body: dict[str, Any] = {}
        if self.disable_thinking:
            # OpenRouter-normalised reasoning control. Harmless on providers that
            # ignore it; turns off Gemini "thinking" where honoured.
            extra_body["reasoning"] = {"enabled": False}
        if extra_body:
            params["extra_body"] = extra_body

        def _call():
            try:
                return self._client.chat.completions.create(**params)
            except (self._openai.RateLimitError, self._openai.APIConnectionError) as exc:
                raise RetryableError(str(exc)) from exc
            except self._openai.APIStatusError as exc:
                if getattr(exc, "status_code", 0) >= 500:
                    raise RetryableError(str(exc)) from exc
                raise

        resp = with_retry(
            _call,
            max_attempts=self.retry_cfg.get("max_attempts", 8),
            base_delay=self.retry_cfg.get("base_delay", 2.0),
            max_delay=self.retry_cfg.get("max_delay", 90.0),
            label=f"openai_compat:{self.model_id}",
        )

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=resp,
        )
