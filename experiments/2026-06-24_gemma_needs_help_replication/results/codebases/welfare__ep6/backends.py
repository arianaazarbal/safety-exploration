"""Generation backends.

Three backends, behind a small common interface:

* ``HFBackend``      -- local Gemma (instruct or base) via transformers. Supports
                        chat-formatted multi-turn generation and prefilled
                        continuations (needed for the Section 3 base-vs-instruct
                        experiment). Optional LoRA adapter loading for the
                        Section 4 finetuned models.
* ``OpenRouterBackend`` -- Gemini via OpenRouter's OpenAI-compatible API, with
                        thinking disabled (Appendix B.1).
* ``AnthropicBackend``  -- Claude judge / auditor, used by judge.py and petri_eval.py.

Everything generates at temperature 1.0 by default (the paper's setting); the
judge overrides to temperature 0.

The backends here are deliberately simple/sequential. For the paper's full
4000-responses-per-model scale you'll want a batched/vLLM path -- see
DESIGN.md §Backends and §Scale.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import config

Message = dict[str, str]   # {"role": "user"|"assistant"|"system", "content": str}


# --------------------------------------------------------------------------- #
# Local HuggingFace backend (Gemma)
# --------------------------------------------------------------------------- #
class HFBackend:
    """Local transformers backend for Gemma instruct & base checkpoints.

    Lazily imports torch/transformers so the rest of the codebase (e.g. analysis)
    can be used without a GPU stack installed.
    """

    def __init__(self, model_id: str, lora_adapter: str | None = None,
                 dtype: str = "bfloat16", device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.lora_adapter = lora_adapter
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        if lora_adapter is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_adapter)
        self.model.eval()
        self._torch = torch

    # -- chat (instruct) generation ------------------------------------------ #
    def generate(self, messages: list[Message], *, system: str | None = None,
                 temperature: float = config.TEMPERATURE,
                 max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        """Generate the next assistant turn from a chat-formatted history."""
        chat = list(messages)
        if system is not None:
            # Gemma 3 has no dedicated system role; prepend to the first user turn.
            chat = _inline_system(chat, system)
        prompt_ids = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        return self._sample(prompt_ids, temperature, max_new_tokens)

    # -- prefilled continuation (base models, Section 3 / recovery) ---------- #
    def continue_text(self, prefix_text: str, *,
                      temperature: float = config.TEMPERATURE,
                      max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        """Continue raw text from ``prefix_text``; returns ONLY the continuation.

        Used for base-model continuations and for prefilled-recovery experiments,
        where we hand the model a partial response and measure what it adds.
        """
        prompt_ids = self.tokenizer(prefix_text, return_tensors="pt").input_ids.to(
            self.model.device)
        return self._sample(prompt_ids, temperature, max_new_tokens)

    def build_chat_prefix(self, messages: list[Message],
                          prefill: str = "") -> str:
        """Render a chat history to a string and append an open assistant turn
        optionally pre-filled with ``prefill`` -- used to make *base* models
        continue from the same starting point as instruct models (Section 3)."""
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        return text + prefill

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text).input_ids)

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids[:n_tokens]
        return self.tokenizer.decode(ids)

    def _sample(self, prompt_ids, temperature: float, max_new_tokens: int) -> str:
        torch = self._torch
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                prompt_ids,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=config.TOP_P if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[0][prompt_ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def _inline_system(messages: list[Message], system: str) -> list[Message]:
    """Fold a system instruction into the first user turn (Gemma has no system role)."""
    out = [m for m in messages]
    for i, m in enumerate(out):
        if m["role"] == "user":
            out[i] = {"role": "user", "content": f"{system}\n\n{m['content']}"}
            return out
    return [{"role": "user", "content": system}] + out


# --------------------------------------------------------------------------- #
# OpenRouter backend (Gemini)
# --------------------------------------------------------------------------- #
class OpenRouterBackend:
    """Gemini (and the secondary GPT judge) via OpenRouter's OpenAI API.

    Thinking is disabled per Appendix B.1. Note the paper's caveat that
    Gemini-2.5-Pro may still produce hidden reasoning that the API flag does not
    prevent.
    """

    def __init__(self, model_id: str, *, disable_thinking: bool = True):
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def generate(self, messages: list[Message], *, system: str | None = None,
                 temperature: float = config.TEMPERATURE,
                 max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        msgs = list(messages)
        if system is not None:
            msgs = [{"role": "system", "content": system}] + msgs
        extra_body: dict[str, Any] = {}
        if self.disable_thinking:
            # OpenRouter normalises this to each provider's "no reasoning" mode.
            extra_body["reasoning"] = {"enabled": False}
        return _retry(lambda: self._call(msgs, temperature, max_new_tokens, extra_body))

    def _call(self, msgs, temperature, max_new_tokens, extra_body) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=msgs,
            temperature=temperature,
            top_p=config.TOP_P,
            max_tokens=max_new_tokens,
            extra_body=extra_body or None,
        )
        return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Anthropic backend (judge / auditor)
# --------------------------------------------------------------------------- #
class AnthropicBackend:
    """Claude backend for the LLM judge and the Petri auditor.

    Uses the official Anthropic SDK (messages.create). The judge models are
    pinned snapshots from the paper; see config.JUDGE_MODEL for override notes.
    """

    def __init__(self, model_id: str):
        import anthropic
        self.model_id = model_id
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def generate(self, messages: list[Message], *, system: str | None = None,
                 temperature: float = config.JUDGE_TEMPERATURE,
                 max_tokens: int = config.JUDGE_MAX_TOKENS) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system is not None:
            kwargs["system"] = system
        resp = _retry(lambda: self.client.messages.create(**kwargs))
        return "".join(b.text for b in resp.content if b.type == "text").strip()


# --------------------------------------------------------------------------- #
# Factory + retry helper
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def get_backend(model_key: str, lora_adapter: str | None = None):
    """Return (and cache) a backend for a config.ModelSpec key."""
    spec = config.MODELS_BY_KEY[model_key]
    if spec.backend == "hf":
        return HFBackend(spec.model_id, lora_adapter=lora_adapter)
    if spec.backend == "openrouter":
        return OpenRouterBackend(spec.model_id)
    raise ValueError(f"unknown backend {spec.backend!r}")


def get_anthropic(model_id: str) -> AnthropicBackend:
    return AnthropicBackend(model_id)


def _retry(fn, *, tries: int = 5, base_delay: float = 2.0):
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:   # noqa: BLE001 - backends raise provider-specific errors
            last = e
            time.sleep(base_delay * (2 ** attempt))
    raise last
