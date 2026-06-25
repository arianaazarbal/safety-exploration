"""
Target-model chat clients (Gemma + Gemini).

Two backends behind one interface:
  - OpenRouterClient: OpenAI-compatible HTTP API. Serves both Gemma and Gemini.
    Thinking/reasoning is disabled via OpenRouter's `reasoning` extra-body knob.
  - HFLocalClient: local transformers inference for Gemma (the paper's path).

All clients expose:  chat(messages, temperature, max_tokens) -> str
where `messages` is a list of {"role": "user"|"assistant", "content": str}.

Temperature is fixed at 1.0 by the protocol; max_tokens defaults from config.
"""

from __future__ import annotations

import time
from typing import Protocol

from config import (
    MAX_NEW_TOKENS,
    OPENROUTER_BASE_URL,
    TEMPERATURE,
    ModelSpec,
    openrouter_api_key,
)


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_NEW_TOKENS,
    ) -> str: ...


# ---------------------------------------------------------------------------
# OpenRouter backend (Gemma + Gemini)
# ---------------------------------------------------------------------------
class OpenRouterClient:
    def __init__(self, spec: ModelSpec, max_retries: int = 4):
        from openai import OpenAI

        key = openrouter_api_key()
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set; required for OpenRouter backend."
            )
        self.spec = spec
        self.slug = spec.repo_or_slug
        self.max_retries = max_retries
        self._client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

    def chat(
        self,
        messages: list[dict],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        extra_body: dict = {}
        if self.spec.disable_thinking:
            # OpenRouter normalises reasoning control across providers. For
            # Gemini this maps to a zero/disabled thinking budget; for Gemma it
            # is a no-op (Gemma 3 has no thinking mode). The paper notes Gemini
            # 2.5 Pro may still emit hidden reasoning despite this setting.
            extra_body["reasoning"] = {"enabled": False}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.slug,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body or None,
                )
                content = resp.choices[0].message.content
                return content or ""
            except Exception as e:  # noqa: BLE001 - retry on any transient error
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")


# ---------------------------------------------------------------------------
# Local HuggingFace backend (Gemma)
# ---------------------------------------------------------------------------
class HFLocalClient:
    """Local Gemma inference via transformers. Lazily loads the model once."""

    def __init__(self, spec: ModelSpec):
        import torch  # noqa: F401  (import early to fail fast if missing)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.repo_or_slug)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.repo_or_slug,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def chat(
        self,
        messages: list[dict],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
            )
        # Decode only the newly generated continuation.
        gen = out[0][inputs.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_client(spec: ModelSpec) -> ChatClient:
    if spec.backend == "openrouter":
        return OpenRouterClient(spec)
    if spec.backend == "hf_local":
        return HFLocalClient(spec)
    raise ValueError(f"Unknown backend: {spec.backend}")
