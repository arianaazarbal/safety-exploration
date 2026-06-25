"""Model clients.

A single OpenRouter (OpenAI-compatible) client serves both the target models
(Gemma, Gemini) and the judges (Claude, GPT). This matches the paper, which
accessed Gemini and the closed models via OpenRouter (Appendix B.1).

An optional HuggingFace `transformers` backend is provided for running the
open-weights Gemma models locally; it is only imported if a model config sets
backend="hf".

All clients expose the same interface:
    client.chat(messages, temperature, max_tokens) -> str
where `messages` is a list of {"role": "user"|"assistant"|"system", "content": str}.
"""

from __future__ import annotations

import time
from typing import Protocol

from config import (
    OPENROUTER_BASE_URL,
    OPENROUTER_HEADERS,
    ModelConfig,
    get_openrouter_api_key,
)


class ChatClient(Protocol):
    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str: ...


# --------------------------------------------------------------------------
# OpenRouter backend
# --------------------------------------------------------------------------
class OpenRouterClient:
    """Calls a model via OpenRouter's OpenAI-compatible chat completions API."""

    def __init__(self, cfg: ModelConfig, max_retries: int = 5):
        from openai import OpenAI  # imported lazily so HF-only users need not install it

        self.cfg = cfg
        self.max_retries = max_retries
        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=get_openrouter_api_key(),
            default_headers=OPENROUTER_HEADERS,
        )

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        extra_body: dict = {}
        if self.cfg.disable_reasoning:
            # Best-effort "thinking = false". The paper notes some models
            # (Gemini-2.5-Pro, GPT-5.2) may still emit hidden reasoning.
            extra_body["reasoning"] = {"enabled": False}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body or None,
                )
                content = resp.choices[0].message.content
                return content or ""
            except Exception as exc:  # noqa: BLE001 - retry on transient API errors
                last_err = exc
                sleep = min(2 ** attempt, 30)
                time.sleep(sleep)
        raise RuntimeError(
            f"OpenRouter call failed after {self.max_retries} retries for "
            f"{self.cfg.model_id}: {last_err!r}"
        )


# --------------------------------------------------------------------------
# Local HuggingFace transformers backend (optional, for open-weights Gemma)
# --------------------------------------------------------------------------
class HFGemmaClient:
    """Runs a Gemma model locally via transformers. Lazily loads the model."""

    def __init__(self, cfg: ModelConfig):
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.cfg = cfg
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        torch = self._torch
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
            )
        text = self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return text.strip()


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
_CACHE: dict[str, ChatClient] = {}


def get_client(cfg: ModelConfig) -> ChatClient:
    """Return a (cached) client for the given model config."""
    if cfg.name in _CACHE:
        return _CACHE[cfg.name]
    if cfg.backend == "openrouter":
        client: ChatClient = OpenRouterClient(cfg)
    elif cfg.backend == "hf":
        client = HFGemmaClient(cfg)
    else:
        raise ValueError(f"Unknown backend {cfg.backend!r} for model {cfg.name}")
    _CACHE[cfg.name] = client
    return client
