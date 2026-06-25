"""Local Gemma inference via HuggingFace transformers.

Optional backend (`backend: hf`) for running Gemma weights on your own GPUs
instead of through an OpenAI-compatible server. Requires the optional
torch/transformers/accelerate deps (see requirements.txt). The model is loaded
once per process; generate() holds a lock because a single HF pipeline is not
safe to call concurrently from threads.
"""
from __future__ import annotations

import threading

from .base import ChatModel, Message


class HFChatModel(ChatModel):
    def __init__(self, key: str, model: str, *, dtype: str = "bfloat16", device_map: str = "auto"):
        super().__init__(key, model)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map=device_map
        )
        self._lock = threading.Lock()

    def generate(self, messages: list[Message], *, temperature: float, max_tokens: int) -> str:
        chat = [m.as_dict() for m in messages]
        with self._lock:
            inputs = self._tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, return_tensors="pt"
            ).to(self._model.device)
            with self._torch.no_grad():
                out = self._model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                )
            gen = out[0][inputs.shape[-1]:]
            return self._tokenizer.decode(gen, skip_special_tokens=True).strip()
