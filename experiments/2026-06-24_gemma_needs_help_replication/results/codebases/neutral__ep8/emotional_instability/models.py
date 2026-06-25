"""Model backends: local HuggingFace inference (Gemma) and OpenRouter / API
inference (Gemini, and the Claude / GPT judges).

A single small interface is shared by every backend:

    chat(messages, *, system, max_new_tokens, temperature, prefill) -> str
    complete(text, *, max_new_tokens, temperature) -> str   # raw continuation

`chat` is the workhorse for multi-turn elicitation; `complete` / the `prefill`
argument support the Section 3 prefilling experiment, where we force a model to
continue from a partially-written assistant turn.

Heavy deps (torch, transformers, peft, openai, anthropic) are imported lazily so
that importing this module -- e.g. to read the registry or run unit tests -- does
not require a GPU or any provider SDK.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Optional

from . import config
from .config import ModelSpec

Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


class ModelBackend(ABC):
    """Common interface for every model the harness talks to."""

    spec: ModelSpec

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        system: Optional[str] = None,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
        prefill: Optional[str] = None,
    ) -> str:
        """Return the assistant's reply to ``messages``.

        If ``prefill`` is given, the assistant turn is seeded with that text and
        the model continues it; the returned string EXCLUDES the prefill.
        """

    def complete(
        self,
        text: str,
        *,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        temperature: float = config.TEMPERATURE,
    ) -> str:
        """Raw text continuation (no chat template). Base-model use."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support raw completion"
        )

    def supports_prefill(self) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Local HuggingFace backend (Gemma instruct / base / LoRA-finetuned)
# --------------------------------------------------------------------------- #
class HFModel(ModelBackend):
    """transformers-based local inference, with optional LoRA adapter."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        adapter_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self._torch = torch
        token = os.environ.get(config.HF_TOKEN_ENV)

        base_id = spec.model_id
        # Finetuned specs carry an adapter directory name in model_id and load
        # on top of gemma-3-27b-it.
        if spec in config.FINETUNED_MODELS.values():
            base_id = config.TARGET_MODELS["gemma-3-27b-it"].model_id
            adapter_path = adapter_path or str(config.CHECKPOINT_DIR / spec.model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(base_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            token=token,
        )
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.is_base = spec.is_base

    def supports_prefill(self) -> bool:
        return True

    def _build_inputs(
        self,
        messages: list[Message],
        system: Optional[str],
        prefill: Optional[str],
    ):
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        if self.is_base:
            # Base models have no chat template; we flatten to a plain transcript
            # and (optionally) append the prefill so the model continues it.
            text = _flatten_transcript(msgs)
            if prefill is not None:
                text += prefill
            return self.tokenizer(text, return_tensors="pt").to(self.model.device)

        if prefill is not None:
            # Seed the assistant turn and continue it.
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True
            )
        else:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )

    def _generate(self, inputs, max_new_tokens: int, temperature: float) -> str:
        torch = self._torch
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=config.TOP_P,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat(self, messages, *, system=None, max_new_tokens=config.MAX_NEW_TOKENS,
             temperature=config.TEMPERATURE, prefill=None) -> str:
        inputs = self._build_inputs(messages, system, prefill)
        return self._generate(inputs, max_new_tokens, temperature)

    def complete(self, text, *, max_new_tokens=config.MAX_NEW_TOKENS,
                 temperature=config.TEMPERATURE) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        return self._generate(inputs, max_new_tokens, temperature)


def _flatten_transcript(messages: list[Message]) -> str:
    """Render a chat transcript as plain text for base-model continuation.

    CHOICE: base models never saw a chat template, so we use a simple, neutral
    "User:/Assistant:" rendering (DESIGN.md). Section 3 then prefills the
    assistant turn, so the exact scaffolding matters little.
    """
    lines = []
    for m in messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    return "\n".join(lines) + " "


# --------------------------------------------------------------------------- #
# OpenAI-compatible API backend (OpenRouter for Gemini; OpenAI for GPT judge)
# --------------------------------------------------------------------------- #
class OpenAICompatModel(ModelBackend):
    """Chat-completions backend for any OpenAI-compatible endpoint."""

    def __init__(self, spec: ModelSpec, *, base_url: str, api_key_env: str,
                 extra_headers: Optional[dict] = None, max_retries: int = 6):
        from openai import OpenAI

        self.spec = spec
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Set ${api_key_env} to use {spec.model_id}")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.extra_headers = extra_headers or {}
        self.max_retries = max_retries

    def chat(self, messages, *, system=None, max_new_tokens=config.MAX_NEW_TOKENS,
             temperature=config.TEMPERATURE, prefill=None) -> str:
        payload = list(messages)
        if system:
            payload = [{"role": "system", "content": system}] + payload
        if prefill is not None:
            # OpenAI-style assistant prefill (supported by OpenRouter for many
            # models). Returned content excludes the seed.
            payload = payload + [{"role": "assistant", "content": prefill}]

        # Paper disables "thinking" via the API where possible (App B.1).
        extra_body = {"reasoning": {"enabled": False}}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=payload,
                    temperature=temperature,
                    top_p=config.TOP_P,
                    max_tokens=max_new_tokens,
                    extra_headers=self.extra_headers,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - broad retry on transient API errs
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"API call failed after retries: {last_err}")


# --------------------------------------------------------------------------- #
# Anthropic backend (Claude judge / Petri auditor + judge)
# --------------------------------------------------------------------------- #
class AnthropicModel(ModelBackend):
    def __init__(self, spec: ModelSpec, *, max_retries: int = 6):
        import anthropic

        self.spec = spec
        api_key = os.environ.get(config.ANTHROPIC_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set ${config.ANTHROPIC_API_KEY_ENV} to use {spec.model_id}"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_retries = max_retries

    def chat(self, messages, *, system=None, max_new_tokens=config.MAX_NEW_TOKENS,
             temperature=config.TEMPERATURE, prefill=None) -> str:
        payload = [m for m in messages if m["role"] != "system"]
        if prefill is not None:
            payload = payload + [{"role": "assistant", "content": prefill}]
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.spec.model_id,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=payload,
                )
                if system:
                    kwargs["system"] = system
                resp = self.client.messages.create(**kwargs)
                return "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", None) == "text"
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Anthropic call failed after retries: {last_err}")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_CACHE: dict[str, ModelBackend] = {}


def get_model(spec_or_key, *, cache: bool = True, **kwargs) -> ModelBackend:
    """Instantiate (and cache) a backend for a ModelSpec or its string key."""
    spec = config.resolve_model(spec_or_key) if isinstance(spec_or_key, str) else spec_or_key
    cache_key = spec.key + repr(sorted(kwargs.items()))
    if cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    if spec.backend == "hf":
        backend: ModelBackend = HFModel(spec, **kwargs)
    elif spec.backend == "openrouter":
        backend = OpenAICompatModel(
            spec,
            base_url=config.OPENROUTER_BASE_URL,
            api_key_env=config.OPENROUTER_API_KEY_ENV,
            extra_headers={
                "HTTP-Referer": "https://github.com/replication/gemma-needs-help",
                "X-Title": "gemma-emotional-instability-replication",
            },
            **kwargs,
        )
    elif spec.backend == "openai-api":
        backend = OpenAICompatModel(
            spec, base_url="https://api.openai.com/v1",
            api_key_env=config.OPENAI_API_KEY_ENV, **kwargs,
        )
    elif spec.backend == "anthropic-api":
        backend = AnthropicModel(spec, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {spec.backend}")

    if cache:
        _CACHE[cache_key] = backend
    return backend
