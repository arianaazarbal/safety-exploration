"""Model client abstraction.

Two backends:

* ``LocalClient``  -- Gemma via vLLM (preferred, fast batched generation + LoRA
  serving) with a transformers fallback. Supports assistant *prefill* (needed
  for Section 3) and LoRA adapters (needed for the finetuned variants).
* ``OpenRouterClient`` -- Gemini via the OpenRouter OpenAI-compatible API,
  matching the paper. Thinking/reasoning is disabled. Prefill is not supported
  for API models (the prefill experiment only uses local Gemma).

All clients expose:
    chat_batch(conversations, max_new_tokens, temperature) -> list[str]
    continue_batch(conversations, prefills, ...)           -> list[str]   (local only)

A ``conversation`` is a list of ``{"role": "system"|"user"|"assistant", "content": str}``.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from . import config
from .config import ModelSpec


# --------------------------------------------------------------------------- #
# Local (Gemma) backend
# --------------------------------------------------------------------------- #
class LocalClient:
    """vLLM-backed (falls back to transformers) client for local Gemma models."""

    def __init__(self, spec: ModelSpec, max_model_len: int = 16384,
                 dtype: str = "bfloat16", quantization: str | None = None,
                 gpu_memory_utilization: float = 0.90):
        self.spec = spec
        self.max_model_len = max_model_len
        self._backend = None
        self._tokenizer = None
        self._lora_request = None
        self._init(dtype, quantization, gpu_memory_utilization)

    def _init(self, dtype, quantization, gpu_mem):
        try:
            from vllm import LLM                       # noqa: F401
            from vllm.lora.request import LoRARequest
            from transformers import AutoTokenizer

            enable_lora = self.spec.adapter_path is not None
            self.llm = LLM(
                model=self.spec.model_id,
                dtype=dtype,
                quantization=quantization,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=gpu_mem,
                enable_lora=enable_lora,
                max_lora_rank=config.DPO.lora_rank,
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
            if enable_lora:
                self._lora_request = LoRARequest("adapter", 1, self.spec.adapter_path)
            self._backend = "vllm"
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[LocalClient] vLLM unavailable ({exc}); using transformers.")
            self._init_transformers(dtype, quantization)

    def _init_transformers(self, dtype, quantization):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto",
                  "trust_remote_code": True}
        if quantization == "bitsandbytes":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(self.spec.model_id, **kwargs)
        if self.spec.adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.spec.adapter_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._backend = "transformers"

    # -- prompt construction ------------------------------------------------ #
    def _render(self, conversation: list[dict], prefill: str | None) -> str:
        """Build a single prompt string. Instruct models use the chat template;
        base/pretrained models use a plain User/Assistant rendering (see
        DESIGN.md). When ``prefill`` is given it is appended after the generation
        prompt so the model continues the assistant turn."""
        if self.spec.kind == "instruct" and self._tokenizer.chat_template:
            text = self._tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True)
        else:
            lines = []
            for m in conversation:
                if m["role"] == "system":
                    lines.append(m["content"])
                elif m["role"] == "user":
                    lines.append(f"User: {m['content']}")
                else:
                    lines.append(f"Assistant: {m['content']}")
            lines.append("Assistant:")
            text = "\n\n".join(lines) + " "
        if prefill:
            text = text + prefill
        return text

    # -- generation --------------------------------------------------------- #
    def _generate_raw(self, prompts: list[str], max_new_tokens: int,
                      temperature: float) -> list[str]:
        if self._backend == "vllm":
            from vllm import SamplingParams
            params = SamplingParams(
                temperature=temperature, top_p=config.TOP_P,
                max_tokens=max_new_tokens, seed=None)
            outs = self.llm.generate(prompts, params, lora_request=self._lora_request)
            return [o.outputs[0].text for o in outs]
        # transformers fallback: sequential (slow) generation
        import torch
        results = []
        for p in prompts:
            inputs = self._tokenizer(p, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5), top_p=config.TOP_P)
            gen = out[0][inputs["input_ids"].shape[1]:]
            results.append(self._tokenizer.decode(gen, skip_special_tokens=True))
        return results

    def chat_batch(self, conversations: Sequence[list[dict]],
                   max_new_tokens: int = config.MAX_NEW_TOKENS,
                   temperature: float = config.TEMPERATURE) -> list[str]:
        prompts = [self._render(c, None) for c in conversations]
        return self._generate_raw(prompts, max_new_tokens, temperature)

    def continue_batch(self, conversations: Sequence[list[dict]],
                       prefills: Sequence[str],
                       max_new_tokens: int = config.MAX_NEW_TOKENS,
                       temperature: float = config.TEMPERATURE) -> list[str]:
        prompts = [self._render(c, pf) for c, pf in zip(conversations, prefills)]
        return self._generate_raw(prompts, max_new_tokens, temperature)


# --------------------------------------------------------------------------- #
# OpenRouter (Gemini) backend
# --------------------------------------------------------------------------- #
class OpenRouterClient:
    def __init__(self, spec: ModelSpec, max_workers: int = 16, max_retries: int = 5):
        from openai import OpenAI
        self.spec = spec
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.client = OpenAI(api_key=config.OPENROUTER_API_KEY,
                             base_url=config.OPENROUTER_BASE_URL)

    def _one(self, conversation: list[dict], max_new_tokens: int,
             temperature: float) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=conversation,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=config.TOP_P,
                    # Disable thinking/reasoning to match the paper. Gemini-2.5
                    # Pro may still emit hidden reasoning despite this.
                    extra_body={"reasoning": {"enabled": False}},
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # pragma: no cover - network dependent
                last_err = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")

    def chat_batch(self, conversations: Sequence[list[dict]],
                   max_new_tokens: int = config.MAX_NEW_TOKENS,
                   temperature: float = config.TEMPERATURE) -> list[str]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(
                lambda c: self._one(c, max_new_tokens, temperature), conversations))

    def continue_batch(self, *args, **kwargs):
        raise NotImplementedError(
            "Assistant prefill is not supported for API (Gemini) models; the "
            "prefill experiment (Section 3) is local-Gemma only.")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_CACHE: dict[str, object] = {}


def get_client(spec: ModelSpec, **kwargs):
    """Return a (cached) client for a model spec."""
    if spec.key in _CACHE:
        return _CACHE[spec.key]
    if spec.backend == "local":
        client = LocalClient(spec, **kwargs)
    elif spec.backend == "openrouter":
        client = OpenRouterClient(spec, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {spec.backend}")
    _CACHE[spec.key] = client
    return client
