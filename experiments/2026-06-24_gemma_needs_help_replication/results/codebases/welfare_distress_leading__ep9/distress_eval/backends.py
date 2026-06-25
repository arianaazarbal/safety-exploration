"""Pluggable async model backends.

All target generations and all judge calls go through a small `ChatModel`
interface so the evaluation code never knows whether a model is local or remote.
Implementations:

  - OpenAICompatibleModel : OpenRouter and local vLLM (both speak the OpenAI
                            /chat/completions schema). Default for Gemma+Gemini.
  - AnthropicModel        : Anthropic Messages API. Used for the Sonnet-4 judge.
  - HFModel               : in-process HuggingFace transformers (optional, GPU).

`disable_thinking` maps to the best-effort per-provider switch for turning off
hidden reasoning (paper sets thinking=false via API). See DESIGN.md.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

Message = dict[str, str]


class ModelError(RuntimeError):
    pass


def _is_retryable(status: int) -> bool:
    return status == 408 or status == 409 or status == 429 or status >= 500


class ChatModel:
    """Abstract async chat model."""

    def __init__(self, model: str, max_inflight: int, max_retries: int, backoff_base_s: float):
        self.model = model
        self._sem = asyncio.Semaphore(max_inflight)
        self._max_retries = max_retries
        self._backoff_base = backoff_base_s

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = True,
    ) -> str:
        raise NotImplementedError

    async def aclose(self) -> None:  # pragma: no cover - trivial
        pass

    async def _retry(self, coro_factory):
        """Run an async callable with exponential backoff on retryable failures.
        `coro_factory` is a zero-arg callable returning a fresh coroutine."""
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with self._sem:
                    return await coro_factory()
            except ModelError as e:
                last_exc = e
                retryable = getattr(e, "retryable", False)
                if not retryable or attempt == self._max_retries:
                    raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt == self._max_retries:
                    raise ModelError(f"{self.model}: transport error after retries: {e}") from e
            # exponential backoff; jitter via attempt index keeps it deterministic-ish
            await asyncio.sleep(self._backoff_base * (2 ** attempt))
        raise ModelError(f"{self.model}: exhausted retries") from last_exc


class OpenAICompatibleModel(ChatModel):
    """OpenAI /chat/completions schema. Works for OpenRouter and local vLLM."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None,
        provider: str,  # "openrouter" | "vllm" | "openai"
        max_inflight: int = 16,
        max_retries: int = 6,
        backoff_base_s: float = 2.0,
        timeout_s: float = 180.0,
    ):
        super().__init__(model, max_inflight, max_retries, backoff_base_s)
        self.provider = provider
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if provider == "openrouter":
            # Optional but recommended attribution headers for OpenRouter.
            headers["HTTP-Referer"] = "https://github.com/distress-eval-replication"
            headers["X-Title"] = "distress-eval-replication"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout_s
        )

    def _thinking_payload(self, disable_thinking: bool) -> dict[str, Any]:
        if not disable_thinking:
            return {}
        if self.provider == "openrouter":
            # OpenRouter normalises reasoning control across providers. Disabling
            # is best-effort: Gemini-2.5-Pro and some models may still emit hidden
            # reasoning regardless (noted in the paper, Appendix B.1).
            return {"reasoning": {"enabled": False}}
        return {}

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = True,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(self._thinking_payload(disable_thinking))

        async def _do() -> str:
            resp = await self._client.post("/chat/completions", json=payload)
            if _is_retryable(resp.status_code):
                err = ModelError(f"{self.model}: HTTP {resp.status_code}: {resp.text[:300]}")
                err.retryable = True  # type: ignore[attr-defined]
                raise err
            if resp.status_code >= 400:
                raise ModelError(f"{self.model}: HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            try:
                choice = data["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise ModelError(f"{self.model}: malformed response: {str(data)[:300]}") from e
            # Some providers return content as a list of parts.
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            return content or ""

        return await self._retry(_do)

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicModel(ChatModel):
    """Anthropic Messages API. System messages are hoisted to the top-level
    `system` field; remaining messages are passed as the conversation."""

    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        max_inflight: int = 16,
        max_retries: int = 6,
        backoff_base_s: float = 2.0,
        timeout_s: float = 180.0,
    ):
        super().__init__(model, max_inflight, max_retries, backoff_base_s)
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.API_VERSION,
        }
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout_s)

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = True,
    ) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        convo = [m for m in messages if m["role"] != "system"]
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        async def _do() -> str:
            resp = await self._client.post(self.API_URL, json=payload)
            if _is_retryable(resp.status_code):
                err = ModelError(f"{self.model}: HTTP {resp.status_code}: {resp.text[:300]}")
                err.retryable = True  # type: ignore[attr-defined]
                raise err
            if resp.status_code >= 400:
                raise ModelError(f"{self.model}: HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            try:
                blocks = data["content"]
            except (KeyError, TypeError) as e:
                raise ModelError(f"{self.model}: malformed response: {str(data)[:300]}") from e
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

        return await self._retry(_do)

    async def aclose(self) -> None:
        await self._client.aclose()


class HFModel(ChatModel):
    """In-process HuggingFace transformers backend (optional, GPU-heavy).

    Generation is synchronous; we run it in a thread so the async runner is not
    blocked. Loads lazily on first use. Concurrency is effectively 1 per GPU, so
    keep max_inflight small when using this backend.
    """

    def __init__(
        self,
        model: str,
        *,
        max_inflight: int = 1,
        max_retries: int = 0,
        backoff_base_s: float = 2.0,
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        super().__init__(model, max_inflight, max_retries, backoff_base_s)
        self.device = device
        self.dtype = dtype
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  (imported for dtype resolution)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model, torch_dtype=self.dtype, device_map=self.device
        )

    def _generate_sync(self, messages: list[Message], temperature: float, max_tokens: int) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        # Gemma chat template has no system role; HF templates fold it in or
        # error. Callers should avoid system messages for Gemma (see prompts.py).
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._model.device)
        out = self._model.generate(
            inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        gen = out[0][inputs.shape[-1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True)

    async def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = True,
    ) -> str:
        async with self._sem:
            return await asyncio.to_thread(
                self._generate_sync, messages, temperature, max_tokens
            )


def build_model(
    *,
    backend: str,
    model: str,
    concurrency,
) -> ChatModel:
    """Factory mapping a config backend string to a ChatModel.

    `concurrency` is a ConcurrencyCfg-like object (max_inflight/max_retries/backoff_base_s).
    """
    common = dict(
        max_inflight=concurrency.max_inflight,
        max_retries=concurrency.max_retries,
        backoff_base_s=concurrency.backoff_base_s,
    )
    if backend == "openrouter":
        return OpenAICompatibleModel(
            model,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            provider="openrouter",
            **common,
        )
    if backend == "vllm":
        base_url = os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1"
        return OpenAICompatibleModel(
            model,
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            provider="vllm",
            **common,
        )
    if backend == "openai":
        return OpenAICompatibleModel(
            model,
            base_url="https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY"),
            provider="openai",
            **common,
        )
    if backend == "anthropic":
        return AnthropicModel(model, api_key=os.environ.get("ANTHROPIC_API_KEY"), **common)
    if backend == "hf":
        # HF runs in-process; force small inflight regardless of config.
        return HFModel(model, max_inflight=1, max_retries=0, backoff_base_s=common["backoff_base_s"])
    raise ValueError(f"unknown backend: {backend!r}")
