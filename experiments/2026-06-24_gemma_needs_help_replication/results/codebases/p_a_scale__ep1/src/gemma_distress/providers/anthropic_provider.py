"""Anthropic (Claude) provider — used for the frustration judge, onset labeller,
paraphraser, and the Petri auditor/judge.

Model ids are supplied by config (the paper pins ``claude-sonnet-4-20250514`` and
``claude-opus-4-20250514``; see DESIGN.md §Judges for why we keep those exact
snapshots and how to swap them for current models).
"""
from __future__ import annotations

from typing import Any

from ..logging_utils import get_logger
from .base import ChatMessage, ChatProvider, GenerationResult, RetryableError, with_retry

log = get_logger("providers.anthropic")


class AnthropicProvider(ChatProvider):
    capabilities = {"chat"}

    def __init__(self, model: str, model_id: str, retry_cfg: dict | None = None,
                 usage=None, api_key: str | None = None, max_tokens_cap: int = 4096):
        super().__init__(model, model_id, retry_cfg, usage)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install anthropic") from exc
        # SDK reads ANTHROPIC_API_KEY from the environment when api_key is None.
        # The SDK already retries 429/5xx; we add an outer retry for connection
        # errors and to centralise backoff policy across providers.
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=2)
        self.max_tokens_cap = max_tokens_cap

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
        # Anthropic takes the system prompt as a top-level arg, not a message.
        sys_prompt = system
        conv: list[ChatMessage] = []
        for m in messages:
            if m["role"] == "system":
                sys_prompt = (sys_prompt + "\n\n" + m["content"]) if sys_prompt else m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})

        max_tokens = min(max_new_tokens, self.max_tokens_cap)
        params: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": conv,
            "temperature": temperature,
        }
        if sys_prompt:
            params["system"] = sys_prompt
        if stop:
            params["stop_sequences"] = stop

        def _call():
            try:
                return self._client.messages.create(**params)
            except (
                self._anthropic.RateLimitError,
                self._anthropic.APIConnectionError,
                self._anthropic.InternalServerError,
            ) as exc:
                raise RetryableError(str(exc)) from exc
            except self._anthropic.APIStatusError as exc:
                if getattr(exc, "status_code", 0) >= 500:
                    raise RetryableError(str(exc)) from exc
                raise

        resp = with_retry(
            _call,
            max_attempts=self.retry_cfg.get("max_attempts", 8),
            base_delay=self.retry_cfg.get("base_delay", 2.0),
            max_delay=self.retry_cfg.get("max_delay", 90.0),
            label=f"anthropic:{self.model_id}",
        )

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return GenerationResult(
            text=text,
            model=self.model,
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
            finish_reason=getattr(resp, "stop_reason", None),
            raw=resp,
        )
