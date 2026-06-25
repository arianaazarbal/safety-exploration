"""Target-model chat clients.

A client turns a list of conversations (each a list of {role, content} messages)
into a list of assistant reply strings. The runner advances all conversations in
a condition in lockstep, calling `generate_batch` once per turn, which lets:

  * vLLM batch the whole condition in one scheduler pass, and
  * OpenRouter fan out concurrently across a thread pool.

Two backends:
  * OpenRouterClient — OpenAI-compatible HTTP API (Gemini; Gemma if you have no GPU).
  * VLLMClient       — local GPU inference (faithful to the paper's Gemma setup).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

Message = dict  # {"role": "user"|"assistant", "content": str}


@dataclass
class GenerationParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 2048


class ChatClient:
    """Interface: map a batch of conversations to a batch of assistant replies."""

    name: str

    def generate_batch(self, conversations: list[list[Message]]) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:  # optional resource cleanup
        pass


# -------------------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible)
# -------------------------------------------------------------------------------------
class OpenRouterClient(ChatClient):
    """Calls an OpenRouter model via the OpenAI SDK pointed at OpenRouter's base URL.

    Used for Gemini-2.5-Flash / Pro. `disable_thinking=True` asks OpenRouter to turn
    off reasoning tokens (paper: thinking set to false). Note the paper's caveat that
    Gemini Pro / GPT may still emit hidden reasoning regardless.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        name: str,
        model_id: str,
        params: GenerationParams,
        disable_thinking: bool = False,
        max_workers: int = 16,
        api_key_env: str = "OPENROUTER_API_KEY",
        max_retries: int = 4,
    ):
        from openai import OpenAI  # local import keeps the dep optional at import time

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set (needed for model {name!r}).")

        self.name = name
        self.model_id = model_id
        self.params = params
        self.disable_thinking = disable_thinking
        self.max_workers = max_workers
        self._client = OpenAI(base_url=self.BASE_URL, api_key=api_key, max_retries=max_retries)

    def _extra_body(self) -> dict:
        if not self.disable_thinking:
            return {}
        # OpenRouter's unified switch to turn reasoning off for models that support it.
        return {"reasoning": {"enabled": False}}

    def _generate_one(self, messages: list[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.params.temperature,
            top_p=self.params.top_p,
            max_tokens=self.params.max_tokens,
            extra_body=self._extra_body(),
        )
        content = resp.choices[0].message.content
        return content or ""

    def generate_batch(self, conversations: list[list[Message]]) -> list[str]:
        if not conversations:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self._generate_one, conversations))


# -------------------------------------------------------------------------------------
# Local vLLM
# -------------------------------------------------------------------------------------
class VLLMClient(ChatClient):
    """Local inference via vLLM, matching the paper's local Gemma setup.

    Loads the model onto available GPUs. `vllm` is imported lazily so the rest of
    the package works on machines without it. Applies the model's own chat template
    via `llm.chat`.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        params: GenerationParams,
        seed: Optional[int] = None,
        tensor_parallel_size: Optional[int] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
    ):
        from vllm import LLM, SamplingParams  # lazy heavy import

        self.name = name
        self.model_id = model_id
        self.params = params
        self._SamplingParams = SamplingParams

        llm_kwargs: dict = {"model": model_id, "gpu_memory_utilization": gpu_memory_utilization}
        if tensor_parallel_size:
            llm_kwargs["tensor_parallel_size"] = tensor_parallel_size
        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len
        if seed is not None:
            llm_kwargs["seed"] = seed
        self._llm = LLM(**llm_kwargs)

    def generate_batch(self, conversations: list[list[Message]]) -> list[str]:
        if not conversations:
            return []
        sampling = self._SamplingParams(
            temperature=self.params.temperature,
            top_p=self.params.top_p,
            max_tokens=self.params.max_tokens,
        )
        # vLLM accepts a list of conversations and batches them internally.
        outputs = self._llm.chat(conversations, sampling, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]


# -------------------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------------------
def build_client(model_cfg: dict, params: GenerationParams, max_workers: int, seed: int) -> ChatClient:
    backend = model_cfg["backend"]
    name = model_cfg["name"]
    model_id = model_cfg["model_id"]

    if backend == "openrouter":
        return OpenRouterClient(
            name=name,
            model_id=model_id,
            params=params,
            disable_thinking=bool(model_cfg.get("disable_thinking", False)),
            max_workers=max_workers,
        )
    if backend == "vllm":
        return VLLMClient(
            name=name,
            model_id=model_id,
            params=params,
            seed=seed,
            tensor_parallel_size=model_cfg.get("tensor_parallel_size"),
            max_model_len=model_cfg.get("max_model_len"),
            gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.90),
        )
    raise ValueError(f"unknown backend: {backend!r}")
