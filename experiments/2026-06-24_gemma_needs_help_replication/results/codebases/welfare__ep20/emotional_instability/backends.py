"""Model backends.

Three kinds of backend, matching the paper's setup (B.1, B.2):
  - VLLMBackend       : local Gemma generation via vLLM (chat, raw completion,
                        prefill-continuation, optional LoRA adapter).
  - OpenRouterBackend : Gemini via the OpenAI-compatible OpenRouter API, thinking
                        disabled.
  - AnthropicBackend  : Claude (judge / onset / paraphrase / auditor) via the
                        Anthropic API.

API keys are read from the environment:
  OPENROUTER_API_KEY, ANTHROPIC_API_KEY.

Heavy SDKs are imported lazily so that, e.g., analysis-only runs don't require
vLLM/torch to be installed.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

Messages = list[dict]            # [{"role": "user"|"assistant"|"system", "content": str}]

DEFAULT_API_CONCURRENCY = 16


def _concurrent_map(fn, items, max_workers):
    """Map `fn` over `items` preserving order, with bounded concurrency."""
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, x): i for i, x in enumerate(items)}
        for fut, i in futures.items():
            results[i] = fut.result()
    return results


def _retry(fn, *, tries=5, base_delay=2.0):
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:       # noqa: BLE001 -- network/rate-limit retries
            last = exc
            time.sleep(base_delay * (2 ** attempt))
    raise last


# ===========================================================================
# vLLM (local Gemma)
# ===========================================================================
class VLLMBackend:
    def __init__(self, model_id: str, lora_path: Optional[str] = None,
                 max_model_len: int = 8192, max_lora_rank: int = 64,
                 dtype: str = "bfloat16", **llm_kwargs):
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.model_id = model_id
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            max_model_len=max_model_len,
            enable_lora=lora_path is not None,
            max_lora_rank=max_lora_rank,
            **llm_kwargs,
        )
        self.lora_request = (
            LoRARequest("adapter", 1, lora_path) if lora_path else None
        )

    def _sampling_params(self, temperature, max_tokens, seed):
        from vllm import SamplingParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              seed=seed)

    def chat(self, conversations: list[Messages], temperature: float,
             max_tokens: int, seed: Optional[int] = None,
             continue_final: bool = False) -> list[str]:
        """Chat-template generation. If `continue_final` is set, the final message
        (an assistant prefix) is continued rather than a new turn started -- used
        for prefill experiments with instruct models."""
        sp = self._sampling_params(temperature, max_tokens, seed)
        outs = self.llm.chat(
            conversations, sp, lora_request=self.lora_request,
            add_generation_prompt=not continue_final,
            continue_final_message=continue_final,
        )
        return [o.outputs[0].text for o in outs]

    def complete(self, texts: list[str], temperature: float, max_tokens: int,
                 seed: Optional[int] = None) -> list[str]:
        """Raw text completion (no chat template) -- used to continue prefilled
        text with *base / pretrained* models that have no chat template."""
        sp = self._sampling_params(temperature, max_tokens, seed)
        outs = self.llm.generate(texts, sp, lora_request=self.lora_request)
        return [o.outputs[0].text for o in outs]


# ===========================================================================
# OpenRouter (Gemini)
# ===========================================================================
class OpenRouterBackend:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model_id: str, concurrency: int = DEFAULT_API_CONCURRENCY):
        from openai import OpenAI
        self.model_id = model_id
        self.concurrency = concurrency
        self.client = OpenAI(base_url=self.BASE_URL,
                             api_key=os.environ["OPENROUTER_API_KEY"])

    def _one(self, messages: Messages, temperature, max_tokens) -> str:
        def call():
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # Disable hidden reasoning where the provider supports it (B.1).
                extra_body={"reasoning": {"enabled": False}},
            )
            return resp.choices[0].message.content or ""
        return _retry(call)

    def chat(self, conversations: list[Messages], temperature: float,
             max_tokens: int, seed: Optional[int] = None,
             continue_final: bool = False) -> list[str]:
        if continue_final:
            raise NotImplementedError(
                "prefill-continuation is not supported for API models (Gemini "
                "has no base model and cannot be prefilled); see DESIGN.md.")
        return _concurrent_map(
            lambda c: self._one(c, temperature, max_tokens),
            conversations, self.concurrency,
        )


# ===========================================================================
# Anthropic (Claude judge / helpers)
# ===========================================================================
class AnthropicBackend:
    def __init__(self, model_id: str, concurrency: int = DEFAULT_API_CONCURRENCY,
                 max_tokens: int = 1024):
        import anthropic
        self.model_id = model_id
        self.concurrency = concurrency
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _one(self, prompt: str, max_tokens: Optional[int]) -> str:
        def call():
            resp = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens or self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in resp.content if b.type == "text")
        return _retry(call)

    def complete_prompts(self, ps: list[str],
                         max_tokens: Optional[int] = None) -> list[str]:
        return _concurrent_map(lambda p: self._one(p, max_tokens), ps,
                               self.concurrency)


# ===========================================================================
# Factory
# ===========================================================================
def make_generation_backend(name: str, cfg: dict, lora_path: Optional[str] = None,
                            models_key: str = "models", **kw):
    """Construct a generation backend for a model named in cfg[models_key]."""
    entry = cfg[models_key][name]
    backend = entry["backend"]
    model_id = entry["model_id"]
    if backend == "vllm":
        return VLLMBackend(model_id, lora_path=lora_path, **kw)
    if backend == "openrouter":
        if lora_path:
            raise ValueError("LoRA adapters apply only to local (vLLM) models.")
        return OpenRouterBackend(model_id, **kw)
    raise ValueError(f"unknown backend {backend!r}")


def make_judge_backend(cfg: dict) -> AnthropicBackend:
    return AnthropicBackend(cfg["judge"]["model_id"])
