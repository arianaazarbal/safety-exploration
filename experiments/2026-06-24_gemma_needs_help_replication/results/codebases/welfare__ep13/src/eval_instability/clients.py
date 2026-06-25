"""Unified LLM client abstraction over the three providers we need:

  * "hf"          - local HuggingFace inference for Gemma (and finetuned Gemma).
  * "openrouter"  - OpenAI-compatible API for Gemini-2.5-flash/-pro and the
                    secondary GPT-5-mini judge (matches the paper's routing).
  * "anthropic"   - Claude judge / Petri auditor & judge.

Every backend exposes the same two methods:

    chat(messages, max_new_tokens, temperature, ...) -> str
    generate_with_prefill(messages, prefill, ...)     -> str   (hf only)

`messages` is a list of {"role": "user"|"assistant"|"system", "content": str}.

The HF backend is loaded lazily so that importing this module never requires a
GPU or network access; the API backends are also lazy. This keeps the code
importable for static checks / smoke tests even without credentials.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import config
from config import ModelSpec


# ==========================================================================
# Base interface
# ==========================================================================
class LLMClient:
    spec: ModelSpec

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        top_p: float = config.TOP_P,
        system: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def supports_prefill(self) -> bool:
        return False


# ==========================================================================
# HuggingFace local backend (Gemma)
# ==========================================================================
class HFClient(LLMClient):
    """Local inference for Gemma instruct/base models and LoRA finetunes.

    For *base* (pretrained) Gemma we do not have a chat template, so the
    prefill path builds a plain-text continuation prompt instead. This is the
    mechanism the paper uses in Section 3 to compare base and instruct models.
    """

    def __init__(
        self,
        spec: ModelSpec,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.spec = spec
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    # -- lazy loading -----------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, self.dtype)
        kwargs = {"torch_dtype": torch_dtype, "device_map": self.device_map}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.spec.model_id, **kwargs)

        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    # -- helpers ----------------------------------------------------------
    def _apply_template(self, messages: list[dict], system: Optional[str], add_generation_prompt: bool):
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        return self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _generate(self, prompt_text: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        import torch

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    # -- public API -------------------------------------------------------
    def chat(self, messages, max_new_tokens=config.MAX_NEW_TOKENS, temperature=config.TEMPERATURE,
             top_p=config.TOP_P, system=None) -> str:
        self._ensure_loaded()
        prompt_text = self._apply_template(messages, system, add_generation_prompt=True)
        return self._generate(prompt_text, max_new_tokens, temperature, top_p)

    def supports_prefill(self) -> bool:
        return True

    def generate_with_prefill(
        self,
        messages: list[dict],
        prefill: str,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        top_p: float = config.TOP_P,
        system: Optional[str] = None,
    ) -> str:
        """Continue the assistant turn from `prefill`.

        Instruct models: render the chat template with a generation prompt and
        append the prefill text, so the model continues from it.
        Base models: build a plain transcript (no chat tokens) and append the
        prefill, matching the paper's base-model prefilling approach.
        """
        self._ensure_loaded()
        if self.spec.is_base:
            transcript = _plain_transcript(messages, system)
            prompt_text = transcript + prefill
        else:
            prompt_text = self._apply_template(messages, system, add_generation_prompt=True) + prefill
        return self._generate(prompt_text, max_new_tokens, temperature, top_p)


def _plain_transcript(messages: list[dict], system: Optional[str]) -> str:
    """A minimal plain-text transcript for base models (no chat special tokens)."""
    lines = []
    if system:
        lines.append(system)
    for m in messages:
        tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
        lines.append(f"{tag}: {m['content']}")
    lines.append("Assistant: ")
    return "\n".join(lines)


# ==========================================================================
# OpenRouter (OpenAI-compatible) backend - Gemini & GPT-5-mini judge
# ==========================================================================
class OpenRouterClient(LLMClient):
    def __init__(self, spec: ModelSpec, disable_thinking: bool = True):
        self.spec = spec
        self.disable_thinking = disable_thinking
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI

        api_key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {config.OPENROUTER_API_KEY_ENV} to call {self.spec.model_id} via OpenRouter."
            )
        self._client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)

    def chat(self, messages, max_new_tokens=config.MAX_NEW_TOKENS, temperature=config.TEMPERATURE,
             top_p=config.TOP_P, system=None) -> str:
        self._ensure_client()
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        extra_body = {}
        if self.disable_thinking:
            # The paper sets thinking=false via the API. OpenRouter exposes this
            # as a reasoning control; Gemini-2.5-Pro may still emit hidden
            # reasoning (noted in Appendix B.1).
            extra_body["reasoning"] = {"enabled": False}

        return _retry(
            lambda: self._client.chat.completions.create(
                model=self.spec.model_id,
                messages=msgs,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_body=extra_body or None,
            ).choices[0].message.content
        )


# ==========================================================================
# Anthropic backend - Claude judge / Petri auditor & judge
# ==========================================================================
class AnthropicClient(LLMClient):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import anthropic

        api_key = os.environ.get(config.ANTHROPIC_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {config.ANTHROPIC_API_KEY_ENV} to call {self.spec.model_id}."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages, max_new_tokens=1024, temperature=0.0, top_p=config.TOP_P,
             system=None) -> str:
        self._ensure_client()
        kwargs = dict(
            model=self.spec.model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
        if system:
            kwargs["system"] = system
        return _retry(lambda: self._client.messages.create(**kwargs).content[0].text)


# ==========================================================================
# Retry wrapper for flaky API calls
# ==========================================================================
def _retry(fn, attempts: int = 5, base_delay: float = 2.0):
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we genuinely want to retry anything transient
            last_exc = exc
            sleep = base_delay * (2 ** i)
            time.sleep(min(sleep, 30))
    raise RuntimeError(f"API call failed after {attempts} attempts: {last_exc}")


# ==========================================================================
# Factory
# ==========================================================================
def build_client(spec: ModelSpec, **kwargs) -> LLMClient:
    if spec.provider == "hf":
        return HFClient(spec, **kwargs)
    if spec.provider == "openrouter":
        return OpenRouterClient(spec, **kwargs)
    if spec.provider == "anthropic":
        return AnthropicClient(spec, **kwargs)
    raise ValueError(f"Unknown provider: {spec.provider}")
