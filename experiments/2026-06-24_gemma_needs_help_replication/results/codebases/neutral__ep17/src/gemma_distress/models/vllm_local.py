"""Local vLLM backend (fast batched sampling).

Preferred for the eval, which needs thousands of temperature-1 samples per
model. Supports LoRA adapters (DPO/SFT finetunes) and assistant prefill.
"""
from __future__ import annotations

from typing import Any

from .base import ChatMessage, GenerationConfig, ModelClient


class VLLMClient(ModelClient):
    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self._LoRARequest = LoRARequest
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        adapter = spec.get("adapter_path")
        self.lora_request = None
        self.llm = LLM(
            model=self.model_id,
            dtype="bfloat16",
            enable_lora=adapter is not None,
            max_lora_rank=spec.get("max_lora_rank", 64),
            tensor_parallel_size=spec.get("tensor_parallel_size", 1),
        )
        if adapter:
            self.lora_request = LoRARequest(self.name, 1, adapter)

    def _sampling_params(self, cfg: GenerationConfig):
        from vllm import SamplingParams

        return SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
            seed=cfg.seed,
        )

    def _render(self, messages: list[ChatMessage], prefill: str | None) -> str:
        if self.is_chat:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)
            text += "\nAssistant:"
        if prefill:
            text = text + prefill
        return text

    def generate_n(self, messages: list[ChatMessage], cfg: GenerationConfig) -> list[str]:
        prompt = self._render(messages, cfg.prefill)
        outs = self.llm.generate(
            [prompt], self._sampling_params(cfg), lora_request=self.lora_request
        )
        return [o.text for o in outs[0].outputs]

    def generate_batch(
        self, batch_messages: list[list[ChatMessage]], cfg: GenerationConfig
    ) -> list[list[str]]:
        """vLLM-native batching: many conversations in one decode call."""
        prompts = [self._render(m, cfg.prefill) for m in batch_messages]
        results = self.llm.generate(
            prompts, self._sampling_params(cfg), lora_request=self.lora_request
        )
        return [[o.text for o in r.outputs] for r in results]

    def complete(self, prompt: str, cfg: GenerationConfig) -> list[str]:
        full = prompt + (cfg.prefill or "")
        outs = self.llm.generate([full], self._sampling_params(cfg), lora_request=self.lora_request)
        return [o.text for o in outs[0].outputs]
