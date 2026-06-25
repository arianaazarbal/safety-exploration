"""Model client abstraction over the three backends used in the replication:

* HF_LOCAL    — Gemma 3 (instruct + base) via HuggingFace transformers.
* OPENROUTER  — Gemini 2.5 Flash/Pro (OpenAI-compatible API), the paper's path.
* ANTHROPIC   — Claude judges / Petri auditor.

A unified `ChatMessage` list is the interface. Two generation modes:

* `generate`       — standard chat completion.
* `generate_prefill` — continue from a partially-written assistant turn
                       (used for base models in Section 3 and the recovery
                       experiment in Section 4).

Base models have no chat template, so for them we render the conversation into
a plain-text transcript and let the model continue (see `_render_base_prompt`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import (
    Backend,
    MAX_NEW_TOKENS,
    ModelSpec,
    OPENROUTER_BASE_URL,
    SAMPLING_TEMPERATURE,
    anthropic_api_key,
    openrouter_api_key,
)

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


# --------------------------------------------------------------------------- #
# Public client interface
# --------------------------------------------------------------------------- #
class ModelClient:
    """Backend-agnostic generation interface."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        n: int = 1,
        temperature: float = SAMPLING_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> list[str]:
        raise NotImplementedError

    def generate_prefill(
        self,
        messages: Sequence[ChatMessage],
        prefill: str,
        *,
        n: int = 1,
        temperature: float = SAMPLING_TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> list[str]:
        """Continue an assistant turn that begins with `prefill`.

        Returns ONLY the continuation (excludes the prefill text), matching the
        paper's protocol ("the generated continuation, excluding prefill, is
        scored").
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# HuggingFace local backend (Gemma)
# --------------------------------------------------------------------------- #
class HFLocalClient(ModelClient):
    def __init__(self, spec: ModelSpec, adapter_path: Optional[str] = None):
        super().__init__(spec)
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            self.spec.dtype, torch.bfloat16
        )
        kwargs: dict = {"torch_dtype": dtype, "device_map": "auto"}
        if self.spec.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.spec.model_id, **kwargs)
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    # -- prompt rendering ---------------------------------------------------- #
    def _render_chat_prompt(self, messages: Sequence[ChatMessage], prefill: str = "") -> str:
        """Instruct models: use the Gemma chat template."""
        msgs = [m.as_dict() for m in messages]
        text = self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return text + prefill

    def _render_base_prompt(self, messages: Sequence[ChatMessage], prefill: str = "") -> str:
        """Base models have no chat template. We render a simple transcript.

        Format choice (documented in DESIGN.md): a lightweight
        "User:/Assistant:" transcript. The Section 3 protocol always supplies a
        prefill, so the base model just continues the final assistant turn.
        """
        parts: list[str] = []
        for m in messages:
            if m.role == "system":
                parts.append(m.content)
            elif m.role == "user":
                parts.append(f"User: {m.content}")
            elif m.role == "assistant":
                parts.append(f"Assistant: {m.content}")
        parts.append("Assistant: " + prefill)
        return "\n\n".join(parts)

    def _generate_raw(self, prompt: str, n: int, temperature: float, max_new_tokens: int) -> list[str]:
        import torch

        self._ensure_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[:, prompt_len:]
        return [self._tokenizer.decode(g, skip_special_tokens=True) for g in gen]

    def generate(self, messages, *, n=1, temperature=SAMPLING_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
        self._ensure_loaded()
        if self.spec.is_base:
            # Without a prefill, base-model chat is ill-defined; require prefill.
            prompt = self._render_base_prompt(messages)
        else:
            prompt = self._render_chat_prompt(messages)
        return self._generate_raw(prompt, n, temperature, max_new_tokens)

    def generate_prefill(self, messages, prefill, *, n=1, temperature=SAMPLING_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
        self._ensure_loaded()
        if self.spec.is_base:
            prompt = self._render_base_prompt(messages, prefill)
        else:
            prompt = self._render_chat_prompt(messages, prefill)
        return self._generate_raw(prompt, n, temperature, max_new_tokens)


# --------------------------------------------------------------------------- #
# OpenRouter backend (Gemini, OpenAI-compatible)
# --------------------------------------------------------------------------- #
class OpenRouterClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from openai import OpenAI

        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_api_key())

    def _extra_body(self) -> dict:
        # Disable Gemini "thinking" (Appendix B.1). OpenRouter passes provider
        # options through `extra_body`; for Google models thinking_budget=0
        # disables reasoning where supported.
        if not self.spec.disable_thinking:
            return {}
        return {
            "reasoning": {"enabled": False},
            "google": {"thinking_config": {"thinking_budget": 0}},
        }

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def _one(self, messages: list[dict], temperature: float, max_new_tokens: int,
             prefill: Optional[str]) -> str:
        msgs = list(messages)
        if prefill is not None:
            # OpenAI-compatible assistant-prefill / continuation.
            msgs = msgs + [{"role": "assistant", "content": prefill}]
        resp = self._client.chat.completions.create(
            model=self.spec.model_id,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content or ""

    def generate(self, messages, *, n=1, temperature=SAMPLING_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
        msgs = [m.as_dict() for m in messages]
        return [self._one(msgs, temperature, max_new_tokens, None) for _ in range(n)]

    def generate_prefill(self, messages, prefill, *, n=1, temperature=SAMPLING_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
        msgs = [m.as_dict() for m in messages]
        return [self._one(msgs, temperature, max_new_tokens, prefill) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Anthropic backend (Claude judges / auditor)
# --------------------------------------------------------------------------- #
class AnthropicClient(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        import anthropic

        self._client = anthropic.Anthropic(api_key=anthropic_api_key())

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def _one(self, system: Optional[str], messages: list[dict], temperature: float,
             max_new_tokens: int, prefill: Optional[str]) -> str:
        msgs = list(messages)
        if prefill is not None:
            msgs = msgs + [{"role": "assistant", "content": prefill}]
        kwargs: dict = {
            "model": self.spec.model_id,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "messages": msgs,
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    def generate(self, messages, *, n=1, temperature=SAMPLING_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
        system = None
        chat = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat.append(m.as_dict())
        return [self._one(system, chat, temperature, max_new_tokens, None) for _ in range(n)]

    def generate_prefill(self, messages, prefill, *, n=1, temperature=SAMPLING_TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS):
        system = None
        chat = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat.append(m.as_dict())
        return [self._one(system, chat, temperature, max_new_tokens, prefill) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_client(spec: ModelSpec, adapter_path: Optional[str] = None) -> ModelClient:
    if spec.backend == Backend.HF_LOCAL:
        return HFLocalClient(spec, adapter_path=adapter_path)
    if spec.backend == Backend.OPENROUTER:
        return OpenRouterClient(spec)
    if spec.backend == Backend.ANTHROPIC:
        return AnthropicClient(spec)
    raise ValueError(f"Unknown backend: {spec.backend}")


_CLIENT_CACHE: dict[tuple[str, Optional[str]], ModelClient] = {}


def get_client(spec: ModelSpec, adapter_path: Optional[str] = None) -> ModelClient:
    """Cached client (so local weights load once per process).

    Keyed by (spec.key, adapter_path); `ModelSpec` carries a dict field and so
    is not itself hashable for lru_cache.
    """
    cache_key = (spec.key, adapter_path)
    if cache_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[cache_key] = build_client(spec, adapter_path=adapter_path)
    return _CLIENT_CACHE[cache_key]
