"""OpenRouter client (OpenAI-compatible) for Gemini generation and the
GPT-5-mini cross-judge.

OpenRouter exposes an OpenAI-compatible Chat Completions endpoint, so we use
the ``openai`` SDK pointed at the OpenRouter base URL. This is a genuinely
different provider from Claude, hence a separate SDK/file from
``anthropic_client``.

Thinking/reasoning is disabled to match the paper ("set thinking to be false
via the API"). For Google models on OpenRouter this is done via the
``reasoning`` request field; we pass ``{"enabled": False}`` and additionally a
zero ``max_tokens`` reasoning budget as a belt-and-suspenders for providers
that ignore ``enabled``. The paper notes Gemini-2.5-Pro may still emit hidden
reasoning that the flag does not prevent.
"""

from __future__ import annotations

from .base import Message, ModelClient
from ..config import ModelSpec
from ..utils.io import parallel_map, retry


class OpenRouterChat(ModelClient):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        max_retries: int = 4,
        disable_reasoning: bool = True,
        request_workers: int = 8,
    ):
        import openai  # lazy import

        self.spec = spec
        self.max_retries = max_retries
        self.disable_reasoning = disable_reasoning
        self.request_workers = request_workers
        self._client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,  # falls back to OPENAI_API_KEY/OPENROUTER_API_KEY via env if None
        )

    def _extra_body(self) -> dict:
        if not self.disable_reasoning:
            return {}
        # OpenRouter unified reasoning controls.
        return {"reasoning": {"enabled": False, "max_tokens": 0}}

    def _one(
        self,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> str:
        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self.spec.identifier,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_body=self._extra_body(),
            )
            return resp.choices[0].message.content or ""

        return retry(_call, max_retries=self.max_retries)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        n: int = 1,
    ) -> list[str]:
        # Issue ``n`` independent requests: Gemini via OpenRouter does not
        # reliably honour the ``n`` parameter, and independent calls give the
        # same sampling distribution at temperature 1.
        if n == 1:
            return [self._one(messages, temperature, max_tokens, top_p)]
        return parallel_map(
            lambda _: self._one(messages, temperature, max_tokens, top_p),
            list(range(n)),
            max_workers=min(self.request_workers, n),
        )
