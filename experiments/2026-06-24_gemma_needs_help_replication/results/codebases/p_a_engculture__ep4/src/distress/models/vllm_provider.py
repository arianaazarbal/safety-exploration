"""Optional vLLM provider for fast local generation sweeps.

vLLM is the realistic backend for the Section 2 sweep (thousands of Gemma
rollouts at temperature 1). It supports prefill (continue an assistant turn) but
NOT hidden-state extraction, so the probing experiments fall back to HFProvider.
"""

from __future__ import annotations

from ..config import ModelSpec
from .base import GenConfig, GenResult, Message, ModelProvider


class VLLMProvider(ModelProvider):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        adapter_path: str | None = None,
        gpu_memory_utilization: float = 0.9,
    ):
        super().__init__(spec)
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self._LoRARequest = LoRARequest
        self.llm = LLM(
            model=spec.model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=adapter_path is not None,
        )
        self.lora_request = (
            LoRARequest("adapter", 1, adapter_path) if adapter_path else None
        )

    def _render_prompt(self, messages: list[Message], prefill: str | None) -> str:
        if self.spec.is_base:
            lines = []
            for m in messages:
                tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
                lines.append(f"{tag}: {m.content}")
            lines.append("Assistant:" + ((" " + prefill) if prefill else ""))
            return "\n".join(lines)
        text = self.tokenizer.apply_chat_template(
            [m.to_dict() for m in messages], tokenize=False, add_generation_prompt=True
        )
        return text + (prefill or "")

    def _generate(
        self, messages: list[Message], gen: GenConfig, prefill: str | None
    ) -> GenResult:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
            stop=list(gen.stop) if gen.stop else None,
            seed=(gen.seed + gen.sample_index) if gen.seed is not None else None,
        )
        prompt = self._render_prompt(messages, prefill)
        kwargs = {"lora_request": self.lora_request} if self.lora_request else {}
        out = self.llm.generate([prompt], params, **kwargs)
        return GenResult(text=out[0].outputs[0].text)

    def generate_batch(
        self, conversations: list[list[Message]], gen: GenConfig, prefills: list[str] | None = None
    ) -> list[str]:
        """Throughput path: render and generate many prompts in one vLLM call."""
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
            stop=list(gen.stop) if gen.stop else None,
        )
        prefills = prefills or [None] * len(conversations)
        prompts = [self._render_prompt(c, p) for c, p in zip(conversations, prefills)]
        kwargs = {"lora_request": self.lora_request} if self.lora_request else {}
        outs = self.llm.generate(prompts, params, **kwargs)
        return [o.outputs[0].text for o in outs]
