"""Model backends.

Two families of model are in scope and they need different machinery:

* **Gemma** (open weight) -> local inference via HuggingFace `transformers`
  (optionally vLLM for throughput). Local access is required for the prefill
  experiment (Section 3), DPO/SFT finetuning (Section 4) and internal-emotion
  probing (Appendix I), none of which are possible through an API.
* **Gemini** (closed weight) -> API inference via OpenRouter (OpenAI-compatible).

All backends expose the same minimal surface:

    backend.chat(messages, temperature=..., max_new_tokens=..., prefill=...) -> str

`messages` is a list of ``{"role": "system"|"user"|"assistant", "content": str}``.
`prefill`, if given, is a partial assistant string the model must continue from
(only meaningful / supported for the local HF backend).

Models are loaded lazily and cached so a single process can sweep many
conditions without re-loading weights.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import config


Message = dict[str, str]


# --------------------------------------------------------------------------- #
# Open-weight (Gemma) backend
# --------------------------------------------------------------------------- #
class HFBackend:
    """Local HuggingFace backend for Gemma checkpoints.

    Supports chat-formatted generation, raw prefill continuation (needed to make
    base/pretrained models continue an assistant turn), and optional LoRA adapter
    loading for evaluating finetuned models.
    """

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: Optional[str] = None,
        is_base: bool = False,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.is_base = is_base
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- prompt construction ------------------------------------------------- #
    def _render(self, messages: list[Message], prefill: Optional[str]) -> str:
        """Render messages to a single prompt string.

        For instruct models we use the tokenizer chat template. Base/pretrained
        models have no chat template, so we fall back to a simple transcript
        format (Section 3 prefills base models, so this path matters).
        """
        if self.is_base or self.tokenizer.chat_template is None:
            return _plain_transcript(messages, prefill)

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            text += prefill
        return text

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: Optional[str] = None,
    ) -> str:
        import torch

        prompt = self._render(messages, prefill)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        completion = self.tokenizer.decode(gen, skip_special_tokens=True)
        # The returned completion is ONLY the newly generated text; the caller
        # is responsible for prepending the prefill if it wants the full turn.
        return completion


def _plain_transcript(messages: list[Message], prefill: Optional[str]) -> str:
    """A minimal non-chat transcript for base models (Section 3.1)."""
    parts = []
    for m in messages:
        role = {"system": "System", "user": "User", "assistant": "Assistant"}[m["role"]]
        parts.append(f"{role}: {m['content']}")
    parts.append("Assistant:" + (f" {prefill}" if prefill else ""))
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# API (Gemini via OpenRouter) backend
# --------------------------------------------------------------------------- #
class OpenRouterBackend:
    """OpenAI-compatible client pointed at OpenRouter, used for Gemini models.

    Per Appendix B.1 the paper disables thinking via the API where possible.
    """

    def __init__(self, model_id: str):
        from openai import OpenAI

        self.model_id = model_id
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.OPENROUTER_API_KEY,
        )

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = config.TEMPERATURE,
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        prefill: Optional[str] = None,
    ) -> str:
        if prefill:
            # OpenRouter/Gemini does not reliably support assistant prefill; the
            # prefill experiment is Gemma-only, so this should never be hit.
            raise NotImplementedError("prefill is only supported on the local HF backend")

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            # Disable reasoning/thinking tokens where the provider honours it.
            extra_body={"reasoning": {"enabled": False}},
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def get_backend(model_key: str, adapter_path: Optional[str] = None):
    """Return a cached backend for a model key from config.MODELS."""
    spec = config.MODELS[model_key]
    if spec.backend == "hf":
        return HFBackend(spec.model_id, adapter_path=adapter_path, is_base=spec.is_base)
    if spec.backend == "openrouter":
        if adapter_path:
            raise ValueError("adapters are only applicable to local HF models")
        return OpenRouterBackend(spec.model_id)
    raise ValueError(f"unknown backend {spec.backend!r}")
