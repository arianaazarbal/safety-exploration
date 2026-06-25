"""Model clients for the models under test (Gemma + Gemini).

Two backends behind one `ModelClient.chat(messages) -> str` interface:

* HFBackend       -- local HuggingFace `transformers` inference (paper-faithful for Gemma).
* OpenRouterBackend -- OpenAI-compatible HTTP API (Gemini, and optionally Gemma).

`messages` is a list of {"role": "user"|"assistant", "content": str}. All generation
uses temperature=1 (paper Section 2.1) and disables model "thinking" where supported.
"""

from __future__ import annotations

import time
from typing import Protocol

import config
from config import ModelSpec


class _Backend(Protocol):
    def generate(self, messages: list[dict], max_new_tokens: int) -> str: ...


# --------------------------------------------------------------------------- #
# Local HuggingFace backend (Gemma)
# --------------------------------------------------------------------------- #
class HFBackend:
    """Lazily loads a local model and generates with chat templating.

    Gemma-3 *-it models are loaded once and cached per model_id. Requires a GPU with
    enough memory for the chosen size (27B/12B); see README/DESIGN for hardware notes.
    """

    _cache: dict = {}

    def __init__(self, model_id: str):
        self.model_id = model_id

    def _load(self):
        if self.model_id in HFBackend._cache:
            return HFBackend._cache[self.model_id]
        import torch
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        # gemma-3 *-it ships as a multimodal checkpoint (Gemma3ForConditionalGeneration),
        # so AutoModelForCausalLM can fail to load it. Try the causal-LM class first, then
        # fall back to the image-text-to-text class (text-only inputs still work).
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto",
            )
        except Exception:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16, device_map="auto",
            )
        model.eval()
        HFBackend._cache[self.model_id] = (tok, model)
        return tok, model

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        import torch

        tok, model = self._load()
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                top_k=config.TOP_K if config.TOP_K > 0 else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][prompt_len:]
        return tok.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# OpenRouter backend (Gemini; optionally Gemma)
# --------------------------------------------------------------------------- #
class OpenRouterBackend:
    def __init__(self, model_id: str, disable_thinking: bool = True):
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        from openai import OpenAI

        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set; required for OpenRouter models.")
        self.client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
        )

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        extra_body = {}
        if self.disable_thinking:
            # OpenRouter unified reasoning control. Note (paper Appendix B.1): Gemini-2.5
            # Pro may still emit hidden reasoning despite this flag.
            extra_body["reasoning"] = {"enabled": False}

        last_err = None
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    max_tokens=max_new_tokens,
                    extra_body=extra_body or None,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # network / rate-limit / transient
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter generation failed after retries: {last_err}")


# --------------------------------------------------------------------------- #
# Unified client
# --------------------------------------------------------------------------- #
class ModelClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        if spec.backend == "hf":
            self.backend: _Backend = HFBackend(spec.model_id)
        elif spec.backend == "openrouter":
            self.backend = OpenRouterBackend(spec.model_id, spec.disable_thinking)
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"Unknown backend: {spec.backend}")

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        return self.backend.generate(messages, max_new_tokens or config.MAX_NEW_TOKENS)
