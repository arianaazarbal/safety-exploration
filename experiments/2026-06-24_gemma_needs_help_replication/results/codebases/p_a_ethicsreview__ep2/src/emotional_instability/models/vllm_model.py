"""Optional high-throughput vLLM backend for Gemma elicitation sampling.

Mirrors HFModel's chat / continuation surface but uses vLLM offline batched
generation. Probing and training are NOT supported here (use the transformers
backend). Selected via `backend: vllm` in configs/models.yaml.
"""
from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec
from ..utils.logging import get_logger
from .base import ChatModel, Generation, Message, SamplingParams

log = get_logger("models.vllm")


class VLLMModel(ChatModel):
    supports_chat = True
    supports_continuation = True

    def __init__(self, spec: ModelSpec):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.name = spec.name
        self.family = spec.family
        self.kind = spec.kind
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        enable_lora = spec.adapter_path is not None
        self.llm = LLM(model=spec.hf_id, dtype=spec.dtype, enable_lora=enable_lora)
        self._lora_request = None
        if enable_lora:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, spec.adapter_path)
        self.has_chat_template = (
            spec.kind != "base" and self.tokenizer.chat_template is not None
        )

    def _params(self, params: SamplingParams):
        from vllm import SamplingParams as VParams

        return VParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_new_tokens,
            seed=params.seed,
        )

    def chat_batch(
        self, batch: Sequence[Sequence[Message]], params: SamplingParams
    ) -> list[Generation]:
        if not self.has_chat_template:
            raise RuntimeError(f"{self.name} is a base model; use continue_text().")
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in conv],
                tokenize=False,
                add_generation_prompt=True,
            )
            for conv in batch
        ]
        outs = self.llm.generate(prompts, self._params(params), lora_request=self._lora_request)
        return [
            Generation(text=o.outputs[0].text, prompt_messages=tuple(conv), finish_reason="stop")
            for o, conv in zip(outs, batch)
        ]

    def chat(self, messages: Sequence[Message], params: SamplingParams) -> Generation:
        return self.chat_batch([messages], params)[0]

    def continue_text_batch(
        self, prefixes: Sequence[str], params: SamplingParams
    ) -> list[Generation]:
        outs = self.llm.generate(list(prefixes), self._params(params),
                                 lora_request=self._lora_request)
        return [Generation(text=o.outputs[0].text, finish_reason="stop") for o in outs]

    def continue_text(self, prefix: str, params: SamplingParams) -> Generation:
        return self.continue_text_batch([prefix], params)[0]
