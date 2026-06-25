"""Gemma backends.

Two implementations:

* ``GemmaVLLM`` — offline batched inference via vLLM. This is the default and
  the only practical option for the ~4000 temperature-1 samples per model that
  Section 2 requires. Supports LoRA adapters (the DPO / SFT variants) and
  assistant-prefill continuation (Section 3). Base ("-pt") checkpoints are
  driven by raw-text continuation rather than the chat template.

* ``GemmaHF`` — a transformers fallback for environments without vLLM, or for
  small smoke tests. Same interface, much slower.

Both reuse a single weights load per model id (and per LoRA adapter), guarded
by a process-wide cache, because the 27B model is expensive to load.
"""

from __future__ import annotations

import threading

from .base import ChatModel, Message, split_system, trailing_prefill

_VLLM_CACHE: dict[str, object] = {}
_HF_CACHE: dict[str, object] = {}
_CACHE_LOCK = threading.Lock()


def _build_chat_text(tokenizer, messages: list[Message], prefill: str | None) -> str:
    """Render messages with Gemma's chat template, optionally leaving the final
    assistant turn open for continuation (prefill)."""
    msgs = list(messages)
    if prefill is not None:
        msgs = msgs + [{"role": "assistant", "content": prefill}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, continue_final_message=True
        )
    else:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return text


class GemmaVLLM(ChatModel):
    def __init__(self, name: str, model_id: str, *, is_base: bool = False,
                 lora_path: str | None = None, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.90):
        self.name = name
        self.model_id = model_id
        self.is_base = is_base
        self.lora_path = lora_path
        self._max_model_len = max_model_len
        self._gpu_mem = gpu_memory_utilization
        self._lora_request = None
        self._llm = None
        self._tokenizer = None

    # vLLM (and its CUDA context) is loaded lazily so that importing this module
    # is cheap and the choice of GPU model is made at call time.
    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        with _CACHE_LOCK:
            if self.model_id not in _VLLM_CACHE:
                _VLLM_CACHE[self.model_id] = LLM(
                    model=self.model_id,
                    enable_lora=True,
                    max_lora_rank=64,  # matches the rank-64 adapters in the paper
                    max_model_len=self._max_model_len,
                    gpu_memory_utilization=self._gpu_mem,
                    dtype="bfloat16",
                )
            self._llm = _VLLM_CACHE[self.model_id]
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.lora_path:
            # id 1 is fine; we only ever attach one adapter per client.
            self._lora_request = LoRARequest("adapter", 1, self.lora_path)

    def generate(self, messages, *, temperature, max_tokens, n=1, stop=None):
        self._ensure_loaded()
        from vllm import SamplingParams

        base_msgs, prefill = trailing_prefill(messages)

        if self.is_base:
            # Pretrained checkpoints are not chat-tuned: continue raw text.
            # We require a prefill for base models (Section 3 always provides
            # one); without it we fall back to concatenating turn contents.
            prompt = _format_base_prompt(base_msgs, prefill)
        else:
            prompt = _build_chat_text(self._tokenizer, base_msgs, prefill)

        params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, n=n,
            stop=stop, seed=None,
        )
        outs = self._llm.generate(
            [prompt], params, lora_request=self._lora_request, use_tqdm=False
        )
        return [c.text for c in outs[0].outputs]


def _format_base_prompt(messages: list[Message], prefill: str | None) -> str:
    """Plain-text rendering for base models: a lightly-marked transcript that a
    pretrained model will continue. Used by the prefilling study, where the
    prefill carries the real signal."""
    sys, rest = split_system(messages)
    lines = []
    if sys:
        lines.append(sys)
    for m in rest:
        tag = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{tag}: {m['content']}")
    lines.append("Assistant:" + (f" {prefill}" if prefill else ""))
    return "\n".join(lines)


class GemmaHF(ChatModel):
    """transformers fallback (slow; for smoke tests / no-vLLM environments)."""

    def __init__(self, name: str, model_id: str, *, is_base: bool = False,
                 lora_path: str | None = None, device: str = "cuda"):
        self.name = name
        self.model_id = model_id
        self.is_base = is_base
        self.lora_path = lora_path
        self.device = device
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        key = f"{self.model_id}::{self.lora_path}"
        with _CACHE_LOCK:
            if key not in _HF_CACHE:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
                )
                if self.lora_path:
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, self.lora_path)
                _HF_CACHE[key] = model
            self._model = _HF_CACHE[key]
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def generate(self, messages, *, temperature, max_tokens, n=1, stop=None):
        self._ensure_loaded()
        import torch

        base_msgs, prefill = trailing_prefill(messages)
        if self.is_base:
            text = _format_base_prompt(base_msgs, prefill)
        else:
            text = _build_chat_text(self._tokenizer, base_msgs, prefill)

        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        do_sample = temperature > 0
        out = self._model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [self._tokenizer.decode(g, skip_special_tokens=True) for g in gen]
