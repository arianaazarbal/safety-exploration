"""Model backends: a unified chat interface over local HF models (Gemma),
OpenRouter API models (Gemini), and the Anthropic API (judge / Petri agents).

A `Message` is a dict {"role": "user"|"assistant"|"system", "content": str}.
All backends expose `.chat(messages, **gen_kwargs) -> str` and, where it makes
sense, `.chat_batch(list_of_messages, ...) -> list[str]` for throughput.
"""
from __future__ import annotations

import os
from typing import Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

Message = dict[str, str]


# ===========================================================================
# Local HuggingFace backend — used for all Gemma models (instruct, base, LoRA).
# ===========================================================================
class HFBackend:
    def __init__(
        self,
        model_id: str,
        adapter_dir: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        is_base: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.is_base = is_base
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad so that generated tokens align at the right edge for batches.
        self.tokenizer.padding_side = "left"

        kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()
        self._torch = torch

    def _render(self, messages: Sequence[Message], prefill: str | None = None) -> str:
        """Render messages to a prompt string.

        Base (pretrained) models are not chat-tuned, so we render the
        conversation as plain alternating text rather than using the chat
        template (Section 3.1). `prefill` seeds the start of the assistant
        turn so base models continue consistently.
        """
        if self.is_base:
            parts = []
            for m in messages:
                tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
                parts.append(f"{tag}: {m['content']}")
            parts.append("Assistant:" + (f" {prefill}" if prefill else ""))
            return "\n".join(parts)

        text = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        if prefill:
            text = text + prefill
        return text

    def chat(self, messages: Sequence[Message], prefill: str | None = None,
             temperature: float = 1.0, max_new_tokens: int = 1024,
             top_p: float = 1.0, **_: object) -> str:
        return self.chat_batch([list(messages)], prefill=prefill,
                               temperature=temperature, max_new_tokens=max_new_tokens,
                               top_p=top_p)[0]

    def chat_batch(self, batch: Sequence[Sequence[Message]], prefill: str | None = None,
                   temperature: float = 1.0, max_new_tokens: int = 1024,
                   top_p: float = 1.0, **_: object) -> list[str]:
        torch = self._torch
        prompts = [self._render(m, prefill=prefill) for m in batch]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        # For prefilled generation the caller wants only the continuation, so we
        # do NOT prepend the prefill here; conversation.py handles stitching.
        return [t.strip() for t in texts]


# ===========================================================================
# OpenRouter backend (OpenAI-compatible) — used for Gemini models.
# ===========================================================================
class OpenRouterBackend:
    def __init__(self, model_id: str, disable_thinking: bool = True):
        from openai import OpenAI

        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def chat(self, messages: Sequence[Message], temperature: float = 1.0,
             max_new_tokens: int = 1024, top_p: float = 1.0, **_: object) -> str:
        extra_body: dict = {}
        if self.disable_thinking:
            # OpenRouter: disable reasoning tokens where the model supports it.
            extra_body["reasoning"] = {"enabled": False}
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=list(messages),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            extra_body=extra_body or None,
        )
        return (resp.choices[0].message.content or "").strip()


# ===========================================================================
# Anthropic backend — used for the judge and the Petri auditor/judge.
# ===========================================================================
class AnthropicBackend:
    def __init__(self, model_id: str):
        import anthropic

        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def chat(self, messages: Sequence[Message], temperature: float = 0.0,
             max_new_tokens: int = 1024, system: str | None = None, **_: object) -> str:
        sys_text = system
        conv = []
        for m in messages:
            if m["role"] == "system":
                sys_text = (sys_text + "\n\n" + m["content"]) if sys_text else m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        resp = self.client.messages.create(
            model=self.model_id,
            system=sys_text or None,
            messages=conv,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return "".join(block.text for block in resp.content if block.type == "text").strip()


# ===========================================================================
# Factory + cache.
# ===========================================================================
_CACHE: dict[str, object] = {}


def get_backend(spec, generation=None, **overrides):
    """Build (and cache) a backend for a ModelSpec or JudgeSpec-like object."""
    cache_key = f"{spec.backend}:{spec.id}:{getattr(spec, 'base_id', None)}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    disable_thinking = getattr(generation, "disable_thinking", True)
    if spec.backend == "hf":
        is_base = getattr(spec, "role", "") == "base"
        adapter = None
        model_id = spec.id
        if getattr(spec, "base_id", None):
            adapter = spec.id
            model_id = spec.base_id
        backend = HFBackend(model_id, adapter_dir=adapter, is_base=is_base, **overrides)
    elif spec.backend == "openrouter":
        backend = OpenRouterBackend(spec.id, disable_thinking=disable_thinking)
    elif spec.backend == "anthropic":
        backend = AnthropicBackend(spec.id)
    else:
        raise ValueError(f"Unknown backend: {spec.backend}")

    _CACHE[cache_key] = backend
    return backend
