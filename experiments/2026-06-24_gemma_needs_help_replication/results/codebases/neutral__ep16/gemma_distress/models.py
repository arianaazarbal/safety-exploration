"""Unified model-client interface across backends.

A `ModelClient` exposes two operations the experiments need:

  * ``chat(messages, **kw)``      -> str   : standard multi-turn chat completion
  * ``continue_text(prompt, prefill, **kw)`` -> str
        Generate a continuation that *starts from* ``prefill`` (the model's own
        partial response). This is what the Section 3 prefill experiment needs
        so that base (non-chat) models can be steered onto the same trajectory.

Only the HF (local) backend supports ``continue_text`` faithfully — API models
(Gemini) cannot be reliably prefilled, which is why the prefill experiment is
Gemma-only (see DESIGN.md).
"""

from __future__ import annotations

import os
import time
from typing import Sequence

from .config import (ANTHROPIC_API_KEY, HF_TOKEN, JUDGE_TEMPERATURE,
                     MAX_NEW_TOKENS, ModelSpec, OPENAI_API_KEY,
                     OPENROUTER_API_KEY, OPENROUTER_BASE_URL,
                     SAMPLING_TEMPERATURE)

Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


class ModelClient:
    """Abstract base. Subclasses implement `_generate` and optionally prefill."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    # -- public API --------------------------------------------------------- #
    def chat(self, messages: Sequence[Message], *, temperature: float | None = None,
             max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        raise NotImplementedError

    def continue_text(self, messages: Sequence[Message], prefill: str, *,
                      temperature: float | None = None,
                      max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        raise NotImplementedError(
            f"{self.spec.key} ({self.spec.backend}) does not support prefilled "
            "continuation; the prefill experiment is restricted to HF models.")

    def supports_prefill(self) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Local HuggingFace models (Gemma)
# --------------------------------------------------------------------------- #
class HFModel(ModelClient):
    """Local transformers model. Handles both instruct (chat-template) and base
    (raw continuation) checkpoints, and optional LoRA adapters from finetuning."""

    def __init__(self, spec: ModelSpec, *, adapter_path: str | None = None,
                 load_in_4bit: bool = False, device_map: str = "auto"):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kw = dict(torch_dtype=torch.bfloat16, device_map=device_map,
                  token=HF_TOKEN)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch

    def supports_prefill(self) -> bool:
        return True

    # -- prompt construction ----------------------------------------------- #
    def _render(self, messages: Sequence[Message], prefill: str | None = None) -> str:
        """Render messages to a prompt string.

        Instruct models use the chat template. Base models (no chat template
        training) get a lightweight plain-text rendering so we can still drive
        them with prefilled assistant turns (Section 3)."""
        if self.spec.is_base:
            # Minimal, neutral formatting for a pretrained model. The prefill
            # experiment supplies the assistant's partial turn directly.
            parts = []
            for m in messages:
                tag = {"user": "User", "assistant": "Assistant",
                       "system": "System"}[m["role"]]
                parts.append(f"{tag}: {m['content']}")
            parts.append("Assistant:")
            text = "\n".join(parts)
            if prefill:
                text += " " + prefill
            return text

        text = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True)
        if prefill:
            text += prefill        # continue the open assistant turn
        return text

    def _generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kw = dict(max_new_tokens=max_new_tokens,
                      pad_token_id=self.tokenizer.eos_token_id)
        if temperature and temperature > 0:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=1.0)
        else:
            gen_kw.update(do_sample=False)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, **gen_kw)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat(self, messages, *, temperature=None, max_new_tokens=MAX_NEW_TOKENS):
        temperature = SAMPLING_TEMPERATURE if temperature is None else temperature
        return self._generate(self._render(messages), temperature, max_new_tokens)

    def continue_text(self, messages, prefill, *, temperature=None,
                     max_new_tokens=MAX_NEW_TOKENS):
        temperature = SAMPLING_TEMPERATURE if temperature is None else temperature
        return self._generate(self._render(messages, prefill),
                              temperature, max_new_tokens)


# --------------------------------------------------------------------------- #
# OpenRouter chat models (Gemini)
# --------------------------------------------------------------------------- #
class OpenRouterModel(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from openai import OpenAI
        self._client = OpenAI(api_key=OPENROUTER_API_KEY,
                              base_url=OPENROUTER_BASE_URL)

    def chat(self, messages, *, temperature=None, max_new_tokens=MAX_NEW_TOKENS):
        temperature = SAMPLING_TEMPERATURE if temperature is None else temperature
        # Disable provider-side reasoning where supported (paper: thinking=False).
        extra = {"reasoning": {"enabled": False}}
        return _retry(lambda: self._client.chat.completions.create(
            model=self.spec.api_id, messages=list(messages),
            temperature=temperature, max_tokens=max_new_tokens,
            extra_body=extra).choices[0].message.content)


# --------------------------------------------------------------------------- #
# Anthropic models (Claude judge / Petri auditor & judge)
# --------------------------------------------------------------------------- #
class AnthropicModel(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        import anthropic
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def chat(self, messages, *, temperature=None, max_new_tokens=MAX_NEW_TOKENS):
        temperature = JUDGE_TEMPERATURE if temperature is None else temperature
        sys = [m["content"] for m in messages if m["role"] == "system"]
        turns = [m for m in messages if m["role"] != "system"]
        return _retry(lambda: self._client.messages.create(
            model=self.spec.api_id, system=("\n".join(sys) or None),
            messages=turns, temperature=temperature,
            max_tokens=max_new_tokens).content[0].text)


# --------------------------------------------------------------------------- #
# OpenAI-compatible models (GPT-5-mini judge validation)
# --------------------------------------------------------------------------- #
class OpenAIModel(ModelClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from openai import OpenAI
        self._client = OpenAI(api_key=OPENAI_API_KEY)

    def chat(self, messages, *, temperature=None, max_new_tokens=MAX_NEW_TOKENS):
        temperature = JUDGE_TEMPERATURE if temperature is None else temperature
        return _retry(lambda: self._client.chat.completions.create(
            model=self.spec.api_id, messages=list(messages),
            temperature=temperature, max_tokens=max_new_tokens
            ).choices[0].message.content)


# --------------------------------------------------------------------------- #
# Factory + retry helper
# --------------------------------------------------------------------------- #
def _retry(fn, tries: int = 5, base_delay: float = 2.0):
    """Exponential-backoff retry for transient API errors / rate limits."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:           # noqa: BLE001 - broad by design for APIs
            last = e
            time.sleep(base_delay * (2 ** i))
    raise RuntimeError(f"API call failed after {tries} retries: {last}")


_HF_CACHE: dict[str, HFModel] = {}


def load_model(spec: ModelSpec, *, adapter_path: str | None = None,
               **hf_kwargs) -> ModelClient:
    """Instantiate a client for a model spec. HF models are cached per process
    (loading 27B weights is expensive); adapters get distinct cache keys."""
    if spec.backend == "hf":
        cache_key = f"{spec.key}:{adapter_path}"
        if cache_key not in _HF_CACHE:
            _HF_CACHE[cache_key] = HFModel(spec, adapter_path=adapter_path,
                                           **hf_kwargs)
        return _HF_CACHE[cache_key]
    if spec.backend == "openrouter":
        return OpenRouterModel(spec)
    if spec.backend == "anthropic":
        return AnthropicModel(spec)
    if spec.backend == "openai":
        return OpenAIModel(spec)
    raise ValueError(f"Unknown backend: {spec.backend}")
