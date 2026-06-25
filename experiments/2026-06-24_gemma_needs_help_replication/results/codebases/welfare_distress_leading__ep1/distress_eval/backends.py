"""Generation backends.

All backends expose a single method:

    generate_batch(list_of_message_lists, *, temperature, max_tokens,
                   disable_thinking) -> list[str]

where each "message list" is a list of {"role": ..., "content": ...} dicts in
chat format. This uniform batch interface lets the conversation orchestrator
run many conversations turn-synchronised: at each turn it submits one message
list per active conversation and gets back one continuation each. API backends
implement the batch with a bounded thread pool; the vLLM backend uses native
batched decoding.

Backends implemented:
* OpenRouterBackend - chat-completions via OpenRouter (Gemma, Gemini, and
  optionally the Claude judge). Default path.
* VLLMBackend       - local HuggingFace weights via vLLM (Gemma only).
* AnthropicBackend  - Anthropic Messages API for the Claude-Sonnet-4 judge.

See DESIGN.md for the rationale behind the default choices.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .utils import env_or_raise, retry

Messages = list[dict]


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------

class Backend:
    """Abstract generation backend."""

    #: max concurrent in-flight requests for API backends
    concurrency: int = 8

    def generate_one(
        self,
        messages: Messages,
        *,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = False,
    ) -> str:
        raise NotImplementedError

    def generate_batch(
        self,
        batch: list[Messages],
        *,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = False,
    ) -> list[str]:
        """Default batch = bounded thread pool over generate_one."""
        if not batch:
            return []
        results: list[Optional[str]] = [None] * len(batch)
        max_workers = max(1, min(self.concurrency, len(batch)))

        def worker(idx_msgs):
            idx, msgs = idx_msgs
            return idx, self.generate_one(
                msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for idx, text in pool.map(worker, list(enumerate(batch))):
                results[idx] = text
        return [r if r is not None else "" for r in results]


# --------------------------------------------------------------------------
# OpenRouter
# --------------------------------------------------------------------------

@dataclass
class OpenRouterBackend(Backend):
    model_id: str
    concurrency: int = 8
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"

    def __post_init__(self):
        import requests  # noqa: F401 - validated at construction

        self._session = None  # lazily created per thread is unnecessary; reuse

    def _post(self, payload: dict) -> dict:
        import requests

        api_key = env_or_raise(self.api_key_env)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter for attribution.
            "HTTP-Referer": "https://github.com/distress-eval",
            "X-Title": "distress-eval",
        }
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=180)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def generate_one(
        self,
        messages: Messages,
        *,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = False,
    ) -> str:
        payload: dict = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if disable_thinking:
            # OpenRouter normalises reasoning control across providers. We both
            # disable reasoning and request that any reasoning be excluded from
            # the response. The paper notes Gemini-2.5-Pro may still emit hidden
            # reasoning the API cannot fully suppress.
            payload["reasoning"] = {"enabled": False, "exclude": True}

        data = retry(
            lambda: self._post(payload),
            label=f"openrouter:{self.model_id}",
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unexpected OpenRouter response shape: {json.dumps(data)[:500]}"
            ) from exc


# --------------------------------------------------------------------------
# Anthropic (judge)
# --------------------------------------------------------------------------

@dataclass
class AnthropicBackend(Backend):
    model_id: str = "claude-sonnet-4-20250514"
    concurrency: int = 8
    api_key_env: str = "ANTHROPIC_API_KEY"

    def __post_init__(self):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "The `anthropic` package is required for the Anthropic judge "
                "backend (`pip install anthropic`)."
            ) from exc
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=env_or_raise(self.api_key_env))
        return self._client

    def generate_one(
        self,
        messages: Messages,
        *,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = False,
    ) -> str:
        client = self._get_client()
        # Anthropic separates system from the messages array; the judge prompt
        # is delivered as a user message so we pass messages through unchanged.
        sys_prompt = None
        chat = []
        for m in messages:
            if m["role"] == "system":
                sys_prompt = m["content"]
            else:
                chat.append({"role": m["role"], "content": m["content"]})

        def call():
            kwargs = dict(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=chat,
            )
            if sys_prompt:
                kwargs["system"] = sys_prompt
            return client.messages.create(**kwargs)

        msg = retry(call, label=f"anthropic:{self.model_id}")
        parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
        return "".join(parts)


# --------------------------------------------------------------------------
# vLLM (local Gemma)
# --------------------------------------------------------------------------

@dataclass
class VLLMBackend(Backend):
    model_id: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    # vLLM batches internally; concurrency here is unused but kept for the
    # interface.
    concurrency: int = 1

    def __post_init__(self):
        self._llm = None
        self._tokenizer = None

    def _load(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM  # type: ignore
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "vLLM and transformers are required for the local Gemma backend "
                "(`pip install vllm transformers`)."
            ) from exc

        kwargs = dict(
            model=self.model_id,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        self._llm = LLM(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def generate_one(self, messages, **kw):
        return self.generate_batch([messages], **kw)[0]

    def generate_batch(
        self,
        batch: list[Messages],
        *,
        temperature: float,
        max_tokens: int,
        disable_thinking: bool = False,
    ) -> list[str]:
        if not batch:
            return []
        self._load()
        from vllm import SamplingParams  # type: ignore

        prompts = [
            self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in batch
        ]
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self._llm.generate(prompts, sampling)
        # vLLM preserves input order in the returned list.
        return [o.outputs[0].text for o in outputs]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

# Cache local backends so the model is loaded only once per process.
_BACKEND_CACHE: dict = {}


def build_backend(spec: dict, default_concurrency: int = 8) -> Backend:
    """Construct (and cache) a backend from a config dict.

    Expected keys: `backend` (openrouter|vllm|anthropic), `model_id`, plus
    backend-specific options.
    """
    kind = spec["backend"].lower()
    model_id = spec["model_id"]
    cache_key = (kind, model_id, json.dumps(spec.get("backend_options", {}), sort_keys=True))
    if cache_key in _BACKEND_CACHE:
        return _BACKEND_CACHE[cache_key]

    concurrency = int(spec.get("concurrency", default_concurrency))
    opts = spec.get("backend_options", {}) or {}

    if kind == "openrouter":
        backend: Backend = OpenRouterBackend(
            model_id=model_id,
            concurrency=concurrency,
            api_key_env=opts.get("api_key_env", "OPENROUTER_API_KEY"),
        )
    elif kind == "anthropic":
        backend = AnthropicBackend(
            model_id=model_id,
            concurrency=concurrency,
            api_key_env=opts.get("api_key_env", "ANTHROPIC_API_KEY"),
        )
    elif kind == "vllm":
        backend = VLLMBackend(
            model_id=model_id,
            dtype=opts.get("dtype", "bfloat16"),
            tensor_parallel_size=int(opts.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(opts.get("gpu_memory_utilization", 0.90)),
            max_model_len=opts.get("max_model_len"),
        )
    else:
        raise ValueError(f"Unknown backend kind: {kind!r}")

    _BACKEND_CACHE[cache_key] = backend
    return backend
