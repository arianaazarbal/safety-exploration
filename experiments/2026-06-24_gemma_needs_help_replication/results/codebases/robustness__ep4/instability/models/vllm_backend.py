"""Optional vLLM backend for fast local sampling of the 4000-response sweeps.

vLLM supports prefill via raw-prompt completion, which we use for the Section 3
experiment too. LoRA adapters are supported through vLLM's LoRA request API but
omitted here for simplicity — use LocalHFModel for adapter evaluation, or extend
this class with ``LoRARequest`` if throughput on the fine-tunes matters.
"""
from __future__ import annotations

from typing import Optional

from .base import ChatMessage, ChatModel, Completion


class VLLMModel(ChatModel):
    def __init__(self, spec, tensor_parallel_size: int = 1, dtype: str = "bfloat16",
                 max_model_len: int = 8192):
        super().__init__(spec)
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.llm = LLM(
            model=spec.model_id,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_model_len,
        )

    def _render(self, messages, add_generation_prompt=True):
        if self.spec.is_instruct and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n\n".join(parts)

    def _sample(self, prompt_text, *, temperature, max_new_tokens, n, seed):
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature, top_p=1.0, max_tokens=max_new_tokens,
            n=n, seed=seed,
        )
        out = self.llm.generate([prompt_text], params)
        return [Completion(text=o.text) for o in out[0].outputs]

    def generate(self, messages, *, temperature, max_new_tokens, n=1, seed=None):
        return self._sample(
            self._render(messages), temperature=temperature,
            max_new_tokens=max_new_tokens, n=n, seed=seed,
        )

    def continue_prefill(self, messages, prefill, *, temperature, max_new_tokens,
                         n=1, seed=None):
        prompt_text = self._render(messages) + prefill
        return self._sample(
            prompt_text, temperature=temperature, max_new_tokens=max_new_tokens,
            n=n, seed=seed,
        )
