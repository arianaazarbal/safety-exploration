"""Model backends for target generation.

Three backends, behind a common interface:
  * VLLMModel        — local Gemma instruct weights via vLLM (fast batched gen).
  * TransformersModel— local Gemma weights via HF transformers; used for base/pt
                       models (Section 3 prefill) and for generating from a model
                       with a LoRA adapter attached.
  * OpenRouterModel  — remote Gemini via an OpenAI-compatible endpoint.

Common interface (all batched — the rollout engine fans out across a whole
category at each turn depth):
    chat_batch(conversations, params)     -> list[str]   # chat-formatted
    complete_batch(prompts, params)        -> list[str]   # raw text continuation
    count_tokens(text)                     -> int          # for prefill truncation

`conversations` is a list of message lists, each message being
``{"role": "user"|"assistant"|"system", "content": str}``.

Heavy deps (torch/vllm/transformers) are imported lazily so that Gemini-only or
judge-only runs do not require a GPU stack.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Sequence

import config
from config import ModelSpec


@dataclass(frozen=True)
class GenParams:
    temperature: float = config.GEN_TEMPERATURE
    top_p: float = config.GEN_TOP_P
    max_tokens: int = config.GEN_MAX_TOKENS
    seed: int = config.GEN_SEED

    def with_seed(self, seed: int) -> "GenParams":
        return replace(self, seed=seed)


Conversation = list[dict]          # list of {"role", "content"}


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
class BaseModel:
    spec: ModelSpec

    def chat_batch(self, conversations: Sequence[Conversation],
                   params: GenParams,
                   seeds: Sequence[int] | None = None) -> list[str]:
        raise NotImplementedError

    def complete_batch(self, prompts: Sequence[str], params: GenParams,
                       per_prompt_seeds: Sequence[int] | None = None) -> list[str]:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    # Convenience single-item wrappers.
    def chat(self, conversation: Conversation, params: GenParams) -> str:
        return self.chat_batch([conversation], params)[0]

    def render_chat_prompt(self, conversation: Conversation,
                           add_generation_prompt: bool = True) -> str:
        """Return the chat-templated prompt string (for prefill continuation)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# vLLM backend (local Gemma instruct)
# --------------------------------------------------------------------------- #
class VLLMModel(BaseModel):
    def __init__(self, spec: ModelSpec, adapter_path: str | None = None,
                 tensor_parallel_size: int | None = None,
                 max_model_len: int = 8192):
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.spec = spec
        self.adapter_path = adapter_path
        self._LoRARequest = LoRARequest
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        tp = tensor_parallel_size or int(os.environ.get("EI_TP_SIZE", "1"))
        self.llm = LLM(
            model=spec.model_id,
            tokenizer=spec.model_id,
            dtype="bfloat16",
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            enable_lora=adapter_path is not None,
            max_lora_rank=config.DPO.lora_rank,
            gpu_memory_utilization=float(os.environ.get("EI_GPU_UTIL", "0.90")),
        )
        self._lora_req = (
            LoRARequest("adapter", 1, adapter_path) if adapter_path else None
        )

    def _sampling(self, params: GenParams, seed: int | None = None):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            seed=seed if seed is not None else params.seed,
        )

    def render_chat_prompt(self, conversation, add_generation_prompt=True) -> str:
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    def chat_batch(self, conversations, params, seeds=None) -> list[str]:
        prompts = [self.render_chat_prompt(c) for c in conversations]
        return self.complete_batch(prompts, params, per_prompt_seeds=seeds)

    def complete_batch(self, prompts, params, per_prompt_seeds=None) -> list[str]:
        if per_prompt_seeds is None:
            sp = self._sampling(params)
            outs = self.llm.generate(list(prompts), sp, lora_request=self._lora_req)
        else:
            # vLLM accepts a list of SamplingParams aligned with prompts.
            sps = [self._sampling(params, s) for s in per_prompt_seeds]
            outs = self.llm.generate(list(prompts), sps, lora_request=self._lora_req)
        return [o.outputs[0].text for o in outs]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------- #
# Transformers backend (Gemma base/pt models + adapter-attached generation)
# --------------------------------------------------------------------------- #
class TransformersModel(BaseModel):
    def __init__(self, spec: ModelSpec, adapter_path: str | None = None,
                 batch_size: int = 8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"      # required for batched generation
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch

    def render_chat_prompt(self, conversation, add_generation_prompt=True) -> str:
        # Base (pt) models have no chat template; the prefill experiment supplies
        # an already-rendered prompt via complete_batch, so this is only used for
        # instruct models here.
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    def chat_batch(self, conversations, params, seeds=None) -> list[str]:
        prompts = [self.render_chat_prompt(c) for c in conversations]
        return self.complete_batch(prompts, params, per_prompt_seeds=seeds)

    def complete_batch(self, prompts, params, per_prompt_seeds=None) -> list[str]:
        torch = self._torch
        outputs: list[str] = []
        prompts = list(prompts)
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            if per_prompt_seeds is not None:
                torch.manual_seed(per_prompt_seeds[i])
            else:
                torch.manual_seed(params.seed + i)
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, do_sample=params.temperature > 0,
                    temperature=params.temperature, top_p=params.top_p,
                    max_new_tokens=params.max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            # Strip the prompt tokens; decode only the continuation.
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            outputs.extend(self.tokenizer.batch_decode(new_tokens,
                                                       skip_special_tokens=True))
        return outputs

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


# --------------------------------------------------------------------------- #
# OpenRouter backend (Gemini)
# --------------------------------------------------------------------------- #
class OpenRouterModel(BaseModel):
    def __init__(self, spec: ModelSpec):
        from openai import OpenAI
        self.spec = spec
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for Gemini targets."
            )
        self.client = OpenAI(api_key=config.OPENROUTER_API_KEY,
                             base_url=config.OPENROUTER_BASE_URL)
        # Lazy import to keep retry policy local.
        from .api_clients import chat_completion_with_retry
        self._chat = chat_completion_with_retry

    def _one(self, args) -> str:
        conversation, params, seed = args
        return self._chat(
            client=self.client, model=self.spec.model_id, messages=conversation,
            temperature=params.temperature, top_p=params.top_p,
            max_tokens=params.max_tokens, seed=seed,
            extra_body=self.spec.extra_body,
        )

    def chat_batch(self, conversations, params, seeds=None) -> list[str]:
        from .utils import threaded_map
        seeds = seeds if seeds is not None else [params.seed] * len(conversations)
        args = [(list(c), params, s) for c, s in zip(conversations, seeds)]
        results = threaded_map(self._one, args,
                               max_workers=config.API_MAX_CONCURRENCY,
                               desc=f"gen:{self.spec.key}")
        # Surface exceptions as empty strings but keep alignment; callers log them.
        return [r if isinstance(r, str) else "" for r in results]

    def complete_batch(self, prompts, params, per_prompt_seeds=None) -> list[str]:
        raise NotImplementedError(
            "Raw-text completion / prefill is not supported for API models "
            "(Gemini has no base model and the chat API cannot resume an "
            "assistant turn). The Section 3 prefill experiment is Gemma-only."
        )

    def count_tokens(self, text: str) -> int:
        # Approximate; only the local prefill path needs exact counts.
        return max(1, len(text) // 4)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def load_model(key: str, adapter_path: str | None = None, **kwargs) -> BaseModel:
    spec = config.MODEL_REGISTRY[key]
    if spec.backend == "vllm":
        return VLLMModel(spec, adapter_path=adapter_path, **kwargs)
    if spec.backend == "transformers":
        return TransformersModel(spec, adapter_path=adapter_path, **kwargs)
    if spec.backend == "openrouter":
        if adapter_path:
            raise ValueError("adapter_path is not applicable to API models")
        return OpenRouterModel(spec)
    raise ValueError(f"Unknown backend: {spec.backend}")
