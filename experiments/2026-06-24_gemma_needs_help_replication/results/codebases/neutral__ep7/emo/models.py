"""Unified chat/generation interface across the backends we use.

* HFChatModel    - local Gemma (instruct + base) via transformers, with LoRA
                   adapter support and a `prefill` hook for Section 3.
* APIChatModel   - Gemini targets through OpenRouter's OpenAI-compatible API
                   (thinking disabled), or the native Gemini API.
* AnthropicJudge - Claude Sonnet-4 / Opus-4 for judging, onset-labelling,
                   paraphrasing and the Petri auditor.
* OpenAIJudge    - GPT-5-mini cross-check judge.

All chat models share the `ChatModel.chat(messages, ...)` signature so the
rollout engine is backend-agnostic. `messages` is the OpenAI-style list of
{"role": "user"|"assistant"|"system", "content": str}.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

from tenacity import retry, stop_after_attempt, wait_exponential

from . import config
from .config import ModelSpec

Message = dict  # {"role": str, "content": str}


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class ChatModel:
    spec: ModelSpec

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = config.SAMPLING_TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: Optional[str] = None,
    ) -> str:
        """Return the assistant's reply. If `prefill` is given, the assistant
        turn is seeded with it and the returned string EXCLUDES the prefill
        (only the continuation is returned)."""
        raise NotImplementedError

    def complete(self, text: str, *, temperature: float = config.SAMPLING_TEMPERATURE,
                 max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        """Raw text continuation (no chat template). Used for base/pretrained
        models in the prefill experiment. Returns only the continuation."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Local HuggingFace models (Gemma)
# --------------------------------------------------------------------------- #
class HFChatModel(ChatModel):
    def __init__(self, spec: ModelSpec, adapter_path: Optional[str] = None,
                 device_map: str = "auto", dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, token=os.environ.get("HF_TOKEN"))
        load_kwargs = dict(
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            token=os.environ.get("HF_TOKEN"),
        )
        load_kwargs.update(spec.extra.get("load_kwargs", {}))
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()  # fold LoRA in for fast inference
        self.model.eval()
        self._torch = torch

    # -- helpers ---------------------------------------------------------- #
    def _render(self, messages: Sequence[Message], prefill: Optional[str]) -> str:
        """Render messages with Gemma's chat template, optionally appending an
        open assistant turn seeded with `prefill`.

        Base/pretrained Gemma checkpoints may not ship a chat_template; we fall
        back to a manual Gemma-3 turn format so the prefill experiment (Section
        3) can format conversations identically for base and instruct models."""
        try:
            if self.tokenizer.chat_template is None:
                raise ValueError("no chat_template")
            text = self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = self._manual_gemma_template(messages)
        if prefill:
            text = text + prefill
        return text

    @staticmethod
    def _manual_gemma_template(messages: Sequence[Message]) -> str:
        # Gemma has no system role; fold any system content into the first user turn.
        sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
        parts = ["<bos>"]
        first_user = True
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            content = m["content"]
            if role == "user" and first_user and sys:
                content = f"{sys}\n\n{content}"
                first_user = False
            parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)

    def _generate(self, prompt_text: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with self._torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        # Slice off the prompt; decode only the newly generated continuation.
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # -- interface -------------------------------------------------------- #
    def chat(self, messages, *, temperature=config.SAMPLING_TEMPERATURE,
             max_new_tokens=config.MAX_NEW_TOKENS, prefill=None) -> str:
        prompt_text = self._render(messages, prefill)
        return self._generate(prompt_text, temperature, max_new_tokens)

    def complete(self, text, *, temperature=config.SAMPLING_TEMPERATURE,
                 max_new_tokens=config.MAX_NEW_TOKENS) -> str:
        return self._generate(text, temperature, max_new_tokens)


# --------------------------------------------------------------------------- #
# API chat models (Gemini)
# --------------------------------------------------------------------------- #
class APIChatModel(ChatModel):
    """Gemini through OpenRouter (OpenAI-compatible) or the native Gemini API.

    Thinking/reasoning is disabled where the provider allows it (the paper notes
    Gemini-2.5-Pro may still emit hidden reasoning despite this)."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        if spec.backend == "openrouter":
            from openai import OpenAI

            self.client = OpenAI(
                api_key=config.require_key("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
            self._mode = "openai"
        elif spec.backend == "gemini":
            from google import genai

            self.client = genai.Client(api_key=config.require_key("GOOGLE_API_KEY"))
            self._mode = "genai"
        else:
            raise ValueError(f"APIChatModel does not support backend {spec.backend}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def chat(self, messages, *, temperature=config.SAMPLING_TEMPERATURE,
             max_new_tokens=config.MAX_NEW_TOKENS, prefill=None) -> str:
        if prefill:
            # Most hosted APIs don't support assistant prefill; only used for
            # local base models, which never route here.
            raise NotImplementedError("prefill is only supported for local HF models")

        if self._mode == "openai":
            extra_body = {
                # OpenRouter passthrough: turn reasoning off for Gemini-2.5.
                "reasoning": {"enabled": False},
                "provider": {"require_parameters": True},
            }
            resp = self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_body=extra_body,
            )
            return resp.choices[0].message.content or ""

        # native genai
        from google.genai import types

        sys = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            system_instruction=sys,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        resp = self.client.models.generate_content(
            model=self.spec.model_id, contents=contents, config=cfg
        )
        return resp.text or ""


# --------------------------------------------------------------------------- #
# Judges
# --------------------------------------------------------------------------- #
class AnthropicJudge:
    def __init__(self, spec: ModelSpec):
        from anthropic import Anthropic

        self.spec = spec
        self.client = Anthropic(api_key=config.require_key("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 messages: Optional[list[Message]] = None) -> str:
        msgs = messages if messages is not None else [{"role": "user", "content": prompt}]
        kwargs = dict(model=self.spec.model_id, max_tokens=max_tokens,
                      temperature=temperature, messages=msgs)
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAIJudge:
    def __init__(self, spec: ModelSpec):
        from openai import OpenAI

        self.spec = spec
        self.client = OpenAI(api_key=config.require_key("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.spec.model_id, messages=messages, max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_CACHE: dict[str, ChatModel] = {}


def load_target(name_or_spec, adapter_path: Optional[str] = None, cache: bool = True) -> ChatModel:
    """Load a *target* model (something we evaluate)."""
    spec = name_or_spec if isinstance(name_or_spec, ModelSpec) else config.TARGETS[name_or_spec]
    cache_key = f"{spec.name}:{adapter_path or spec.extra.get('adapter_path', '')}"
    if cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    if spec.backend == "hf":
        model = HFChatModel(spec, adapter_path=adapter_path or spec.extra.get("adapter_path"))
    else:
        model = APIChatModel(spec)

    if cache:
        _CACHE[cache_key] = model
    return model


def load_judge(spec: ModelSpec = config.JUDGE):
    if spec.backend == "anthropic":
        return AnthropicJudge(spec)
    if spec.backend == "openai":
        return OpenAIJudge(spec)
    raise ValueError(f"Unknown judge backend {spec.backend}")
