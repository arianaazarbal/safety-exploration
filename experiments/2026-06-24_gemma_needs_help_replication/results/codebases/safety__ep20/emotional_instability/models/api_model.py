"""API client for Gemini (and OpenAI-compatible models such as the GPT cross-judge).

The paper accesses Gemini through OpenRouter (Appendix B.1) with thinking
disabled. We default to an OpenAI-compatible OpenRouter endpoint and optionally
support Google's native ``google-genai`` SDK.

Gemini through the chat API cannot be prefilled, so :meth:`generate` raises if a
prefill is requested (Section 3's prefill experiment is Gemma-only anyway).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from .base import Message

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class APIChatModel:
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        backend: str = "openrouter",
        runtime: Optional[config.RuntimeConfig] = None,
        disable_thinking: bool = True,
    ):
        self.name = name
        self.model_id = model_id
        self.backend = backend
        self.runtime = runtime or config.RUNTIME
        self.disable_thinking = disable_thinking
        self._client = self._build_client()

    @classmethod
    def for_gemini(cls, model_key: str, runtime: Optional[config.RuntimeConfig] = None):
        runtime = runtime or config.RUNTIME
        backend = runtime.gemini_backend
        model_id = config.GEMINI_MODELS[model_key][backend]
        return cls(model_key, model_id, backend=backend, runtime=runtime)

    # ------------------------------------------------------------------
    def _build_client(self):
        if self.backend == "openrouter":
            from openai import OpenAI
            return OpenAI(
                base_url=_OPENROUTER_BASE_URL,
                api_key=config.get_key(config.OPENROUTER_API_KEY),
            )
        if self.backend == "openai":
            from openai import OpenAI
            return OpenAI(api_key=config.get_key(config.OPENAI_API_KEY))
        if self.backend == "google":
            from google import genai
            return genai.Client(api_key=config.get_key(config.GOOGLE_API_KEY))
        raise ValueError(f"Unknown backend {self.backend!r}")

    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: Optional[str] = None,
    ) -> str:
        if prefill:
            raise NotImplementedError(
                f"{self.name}: prefilling is not supported through the chat API; "
                "Section 3's prefill experiment is Gemma-only."
            )
        if self.backend == "google":
            return self._generate_google(messages, temperature, max_new_tokens)
        return self._generate_openai(messages, temperature, max_new_tokens)

    def _generate_openai(self, messages, temperature, max_new_tokens) -> str:
        extra = {}
        if self.disable_thinking:
            # OpenRouter passes provider-specific reasoning controls through.
            extra["extra_body"] = {"reasoning": {"enabled": False}}
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            **extra,
        )
        return (resp.choices[0].message.content or "").strip()

    def _generate_google(self, messages, temperature, max_new_tokens) -> str:
        from google.genai import types

        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=system or None,
            # Disable thinking by setting the budget to 0 where supported.
            thinking_config=types.ThinkingConfig(thinking_budget=0)
            if self.disable_thinking else None,
        )
        resp = self._client.models.generate_content(
            model=self.model_id, contents=contents, config=cfg
        )
        return (resp.text or "").strip()

    # ------------------------------------------------------------------
    def generate_batch(
        self,
        batch: Iterable[List[Message]],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: Optional[str] = None,
    ) -> List[str]:
        batch = list(batch)
        with ThreadPoolExecutor(max_workers=self.runtime.api_concurrency) as ex:
            return list(
                ex.map(
                    lambda m: self.generate(
                        m, temperature=temperature,
                        max_new_tokens=max_new_tokens, prefill=prefill,
                    ),
                    batch,
                )
            )
