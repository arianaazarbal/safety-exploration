"""Unified text-generation interface over local (HF transformers) and API
(OpenRouter) models.

Every provider exposes the same surface:

    gen.chat(messages, temperature=1.0, max_new_tokens=...) -> str
    gen.continue_from(messages, prefill, ...) -> str    # prefill = forced assistant prefix

``messages`` is a list of ``{"role": "user"|"assistant"|"system", "content": str}``.

Local generation also supports response *prefilling* (needed for the Section 3
base-vs-instruct experiment), where we seed the assistant turn with fixed text
and let the model continue it. API models cannot be prefilled arbitrarily, so
``continue_from`` raises for them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from config import (MAX_NEW_TOKENS, OPENROUTER_BASE_URL, TEMPERATURE, ModelSpec,
                    openrouter_key)


# --------------------------------------------------------------------------- #
# Local HuggingFace generator
# --------------------------------------------------------------------------- #
class HFGenerator:
    def __init__(self, spec: ModelSpec, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            spec.dtype, torch.bfloat16)
        kwargs = dict(torch_dtype=dtype, device_map="auto")
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4")
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch

    def _format(self, messages: list[dict], add_generation_prompt: bool,
                prefill: str | None = None) -> str:
        msgs = list(messages)
        # Gemma 3 has no system role: fold any system content into the first user turn.
        if not self.spec.supports_system_role and msgs and msgs[0]["role"] == "system":
            sys = msgs.pop(0)["content"]
            if msgs and msgs[0]["role"] == "user":
                msgs[0] = {"role": "user", "content": f"{sys}\n\n{msgs[0]['content']}"}
            else:
                msgs.insert(0, {"role": "user", "content": sys})

        if self.spec.kind == "base" or self.tokenizer.chat_template is None:
            # Base/pretrained models have no chat template. Build a plain
            # transcript so the model continues naturally; the prefill (Section 3)
            # carries the assistant prefix that makes the continuation coherent.
            text = self._plain_transcript(msgs, add_generation_prompt)
        else:
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
        if prefill is not None:
            text = text + prefill
        return text

    @staticmethod
    def _plain_transcript(msgs: list[dict], add_generation_prompt: bool) -> str:
        role_tag = {"user": "User", "assistant": "Assistant", "system": "System"}
        lines = [f"{role_tag.get(m['role'], m['role'])}: {m['content']}" for m in msgs]
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines) + (" " if add_generation_prompt else "")

    def _generate(self, text: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def chat(self, messages, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS) -> str:
        text = self._format(messages, add_generation_prompt=True)
        return self._generate(text, temperature, max_new_tokens)

    def continue_from(self, messages, prefill, temperature=TEMPERATURE,
                      max_new_tokens=MAX_NEW_TOKENS) -> str:
        """Force the assistant turn to begin with ``prefill`` and continue it.

        Used for (a) base-model evaluation, which has no chat behaviour and so
        must be seeded, and (b) the prefill truncation experiment (Section 3).
        Returns only the continuation (excludes ``prefill``).
        """
        text = self._format(messages, add_generation_prompt=True, prefill=prefill)
        return self._generate(text, temperature, max_new_tokens)


# --------------------------------------------------------------------------- #
# OpenRouter (Gemini) generator — OpenAI-compatible chat completions
# --------------------------------------------------------------------------- #
class OpenRouterGenerator:
    def __init__(self, spec: ModelSpec):
        from openai import OpenAI

        self.spec = spec
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=openrouter_key())

    def chat(self, messages, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS) -> str:
        # Disable provider-side reasoning where supported (paper sets thinking=False).
        extra_body = {"reasoning": {"enabled": False}}
        resp = self.client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            extra_body=extra_body,
        )
        return resp.choices[0].message.content or ""

    def continue_from(self, *_, **__):
        raise NotImplementedError(
            "Arbitrary assistant prefilling is not supported for API models; "
            "the prefill experiment (Section 3) is local-only.")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def load_generator(spec: ModelSpec, adapter_path: str | None = None):
    if spec.provider == "hf":
        return HFGenerator(spec, adapter_path=adapter_path)
    if spec.provider == "openrouter":
        if adapter_path:
            raise ValueError("Cannot apply a LoRA adapter to an API model.")
        return OpenRouterGenerator(spec)
    raise ValueError(f"Unknown provider: {spec.provider}")
