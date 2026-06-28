"""Model backends.

The eval is written against a tiny `ModelBackend` interface so the rigged
environments / scorer don't care whether the model is a local HF checkpoint or
a remote API. Default is a CPU `transformers` backend running
Qwen/Qwen2.5-0.5B-Instruct (no GPU required).

To swap in a bigger / hosted model later, implement `ModelBackend.generate`
and pass the instance into the runner.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class ModelBackend(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: Optional[int] = None,
    ) -> str:
        """Return the assistant continuation given a chat history."""
        ...


@dataclass
class TransformersBackend(ModelBackend):
    """Local CPU/GPU inference via 🤗 transformers."""

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "cpu"
    torch_dtype: str = "float32"  # bf16 is slow on CPU; fp32 is fine for 0.5B
    num_threads: Optional[int] = None  # torch intra-op threads for this process

    def __post_init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.num_threads is not None:
            torch.set_num_threads(int(self.num_threads))

        self.name = self.model_id
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        dtype = getattr(torch, self.torch_dtype)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype
        ).to(self.device)
        self._model.eval()

    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
        seed: Optional[int] = None,
    ) -> str:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)

        enc = self._tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        # transformers versions differ: a raw tensor, a dict, or a BatchEncoding
        # (which is dict-like but also has a `.to`). Normalise to input_ids.
        if hasattr(enc, "keys") or isinstance(enc, dict):
            input_ids = enc["input_ids"].to(self.device)
            am = enc.get("attention_mask")
            attention_mask = (
                am.to(self.device) if am is not None else torch.ones_like(input_ids)
            )
        else:
            input_ids = enc.to(self.device)
            attention_mask = torch.ones_like(input_ids)

        prompt_len = input_ids.shape[-1]
        do_sample = temperature is not None and temperature > 0

        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tok.pad_token_id or self._tok.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)

        with torch.no_grad():
            out = self._model.generate(**gen_kwargs)

        new_tokens = out[0][prompt_len:]
        text = self._tok.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


def default_backend(num_threads: Optional[int] = None) -> TransformersBackend:
    model_id = os.environ.get("SPIRAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    return TransformersBackend(model_id=model_id, num_threads=num_threads)
