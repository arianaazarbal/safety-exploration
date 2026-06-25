"""OpenAI-compatible client used for API models.

The paper accesses Gemini, Claude (judge) and GPT (validation/Petri) through
OpenRouter (Appendix B.1). OpenRouter exposes the OpenAI Chat Completions API,
so a single client serves all of them; only the model id differs.

API models generally cannot continue an arbitrary *assistant* prefill, so
``continue_prefill`` raises ``PrefillUnsupported`` -- the prefill study (Section
3) is run on local Gemma only, which is consistent with the paper (Gemini has no
public base model to compare against).
"""

from __future__ import annotations

import os

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .base import ChatModel, GenerationConfig, Message

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterModel(ChatModel):
    def __init__(
        self,
        name: str,
        api_id: str,
        family: str = "unknown",
        is_instruct: bool = True,
        max_retries: int = 6,
        base_url: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
    ):
        super().__init__(name=name, family=family, is_instruct=is_instruct)
        # Imported lazily so the package imports cleanly without `openai`.
        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"{api_key_env} is not set; required for API model '{name}'."
            )
        self.api_id = api_id
        self.max_retries = max_retries
        self._client = OpenAI(
            base_url=base_url or os.environ.get("OPENROUTER_BASE_URL", _OPENROUTER_BASE_URL),
            api_key=api_key,
        )

    def _extra_body(self, gen: GenerationConfig) -> dict:
        """Disable provider-side reasoning where supported (Appendix B.1)."""
        if gen.thinking:
            return {}
        # OpenRouter forwards a unified `reasoning` field to providers that
        # support it (Gemini, GPT). Some providers ignore it; that is the
        # paper's documented caveat for Gemini-2.5-Pro / GPT-5.2.
        return {"reasoning": {"enabled": False}}

    def chat(self, messages: list[Message], gen: GenerationConfig) -> str:
        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )
        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self.api_id,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=gen.temperature,
                max_tokens=gen.max_tokens,
                top_p=gen.top_p,
                stop=list(gen.stop) or None,
                extra_body=self._extra_body(gen),
            )
            content = resp.choices[0].message.content
            return content or ""

        return _call()
