"""Local HuggingFace transformers backend for Gemma (optional, GPU-heavy).

This is the closest path to the paper's likely setup (local Gemma weights) but
requires substantial GPU memory for Gemma-3-27B. Kept lazy/optional: nothing
imports torch/transformers unless this backend is actually instantiated.

The model is loaded once per process and cached by model_id, so reuse across
many rollouts is cheap. Generation uses the model's chat template; sampling is
enabled to honour temperature == 1 (paper setting).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from .base import Message


@lru_cache(maxsize=4)
def _load(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return tok, model


class HFLocalClient:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def chat(self, messages: List[Message], *, temperature: float, max_tokens: int) -> str:
        import torch

        tok, model = _load(self.model_id)
        # Gemma chat template has no system role; fold any system text into the
        # first user turn to stay template-compatible.
        norm: List[Message] = []
        for m in messages:
            if m["role"] == "system":
                norm.append({"role": "user", "content": m["content"]})
            else:
                norm.append(m)
        inputs = tok.apply_chat_template(
            norm, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=1.0,
            )
        gen = out[0][inputs.shape[-1]:]
        return tok.decode(gen, skip_special_tokens=True)
