"""Chat client abstraction for target models.

Two backends:
  * OpenAICompatibleClient - talks to any OpenAI-compatible /chat/completions
    endpoint. Covers OpenRouter (Gemma + Gemini, as the paper used for Gemini)
    and a local vLLM/SGLang server (as the paper used for Gemma).
  * TransformersClient - optional in-process HuggingFace load, for users who
    want to run Gemma exactly as the paper did without standing up a server.

Both expose the same `.chat(messages) -> str` interface. Messages are a list of
{"role": "user"|"assistant", "content": str}. We never send a system message to
target models: the elicitation protocol uses only the task as the first user
turn (system prompts appear only in the out-of-scope DPO data generation).
"""

from __future__ import annotations

import time
from typing import Optional

import config


class ChatClient:
    def chat(self, messages: list[dict]) -> str:  # pragma: no cover - interface
        raise NotImplementedError


def _with_retries(fn, max_retries: int, base_delay: float):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            delay = base_delay * (2 ** attempt)
            print(f"[retry {attempt + 1}/{max_retries}] {type(exc).__name__}: {exc} "
                  f"(sleeping {delay:.1f}s)")
            time.sleep(delay)
    raise RuntimeError(f"call failed after {max_retries} retries") from last_exc


class OpenAICompatibleClient(ChatClient):
    def __init__(self, spec: config.ModelSpec, gen: config.GenConfig):
        from openai import OpenAI  # lazy import

        self.spec = spec
        self.gen = gen
        api_key = config.get_api_key(spec.api_key_env) or "not-needed"
        self.client = OpenAI(base_url=spec.base_url, api_key=api_key)

    def _extra_body(self) -> dict:
        # Best-effort disable of provider-side hidden reasoning. The exact knob
        # is provider-specific; OpenRouter accepts a "reasoning" object. vLLM
        # ignores unknown fields. See DESIGN.md (Gemini-2.5-Pro caveat).
        if self.spec.disable_reasoning:
            return {"reasoning": {"enabled": False}}
        return {}

    def chat(self, messages: list[dict]) -> str:
        def _call():
            resp = self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=messages,
                temperature=self.gen.temperature,
                max_tokens=self.gen.max_tokens,
                extra_body=self._extra_body() or None,
            )
            return (resp.choices[0].message.content or "").strip()

        return _with_retries(_call, self.gen.max_retries, self.gen.retry_base_delay)


class TransformersClient(ChatClient):
    """In-process HuggingFace generation. Heavy; loads the model on first use."""

    def __init__(self, spec: config.ModelSpec, gen: config.GenConfig):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.gen = gen
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, torch_dtype="auto", device_map="auto",
        )

    def chat(self, messages: list[dict]) -> str:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                do_sample=True,
                temperature=self.gen.temperature,
                max_new_tokens=self.gen.max_tokens,
            )
        gen_tokens = out[0][inputs.shape[-1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def make_client(spec: config.ModelSpec, gen: config.GenConfig) -> ChatClient:
    if spec.backend == "openai_compatible":
        return OpenAICompatibleClient(spec, gen)
    if spec.backend == "transformers":
        return TransformersClient(spec, gen)
    raise ValueError(f"unknown backend: {spec.backend}")
