"""vLLM backend for high-throughput local Gemma generation.

Section 2 requires ~4000 multi-turn rollouts per model, and the calm-data
generation in Section 4 needs thousands more. vLLM's batched generation makes
this tractable. We expose batch generation and assistant prefill (vLLM supports
continuing a prompt string, which is all prefill needs).

A finetuned LoRA adapter (Section 4) can be served by passing ``lora_path``.
"""

from __future__ import annotations

from .base import ChatBackend, ChatMessage, GenResult
from ..config import GenConfig


class VLLMBackend(ChatBackend):
    supports_prefill = True

    def __init__(self, spec, lora_path: str | None = None, max_model_len: int = 16384, **kwargs):
        super().__init__(spec)
        from vllm import LLM
        from vllm.lora.request import LoRARequest  # noqa: F401  (imported for type/availability)
        from transformers import AutoTokenizer

        self.lora_path = lora_path
        self._LoRARequest = LoRARequest
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.llm = LLM(
            model=spec.model_id,
            dtype="bfloat16",
            max_model_len=max_model_len,
            enable_lora=lora_path is not None,
            max_lora_rank=64,                # matches the rank-64 adapters in Table 9
            **kwargs,
        )
        self.is_instruct = spec.is_instruct

    def _sampling_params(self, gen: GenConfig):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
            seed=gen.seed,
        )

    def _lora_request(self):
        if self.lora_path is None:
            return None
        return self._LoRARequest("adapter", 1, self.lora_path)

    def _render(self, messages: list[ChatMessage], prefill: str | None = None) -> str:
        if self.is_instruct and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            parts = [f"{m['role']}: {m['content']}" for m in messages]
            parts.append("assistant:")
            text = "\n".join(parts)
        if prefill:
            text += prefill
        return text

    # ------------------------------------------------------------------ #
    def generate(self, messages: list[ChatMessage], gen: GenConfig) -> GenResult:
        return self.generate_batch([messages], gen)[0]

    def generate_batch(self, batch, gen) -> list[GenResult]:
        prompts = [self._render(m) for m in batch]
        outs = self.llm.generate(
            prompts, self._sampling_params(gen), lora_request=self._lora_request()
        )
        return [GenResult(text=o.outputs[0].text, completion_tokens=len(o.outputs[0].token_ids)) for o in outs]

    def generate_prefill(self, messages, prefill, gen) -> GenResult:
        prompt = self._render(messages, prefill=prefill)
        outs = self.llm.generate([prompt], self._sampling_params(gen), lora_request=self._lora_request())
        # vLLM returns only the continuation after the prompt (which includes prefill),
        # so this is the continuation excluding prefill.
        return GenResult(text=outs[0].outputs[0].text, completion_tokens=len(outs[0].outputs[0].token_ids))
