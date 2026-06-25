"""Optional high-throughput local backend using vLLM.

The Section 2 sweeps sample 4000 multi-turn responses per model; on a 27B model the
HuggingFace ``generate`` loop is the bottleneck. vLLM's paged-attention engine samples
these far faster and supports ``n`` natively. This backend is selected by setting a
model's ``backend`` to ``vllm`` in config. It is optional — importing this module without
vLLM installed raises a clear error only when the backend is actually constructed.

LoRA adapters are supported through vLLM's ``LoRARequest`` mechanism; prefilling is
implemented by rendering the chat prompt and appending the assistant prefix, identical in
spirit to the HF backend.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import ChatModel, Conversation
from .hf_backend import _gemma_format

logger = logging.getLogger(__name__)


class VLLMBackend(ChatModel):
    """vLLM-backed Gemma model."""

    supports_prefill = True

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
    ):
        super().__init__(name)
        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "vLLM backend requested but vllm is not installed. "
                "Install with `pip install vllm`, or use backend: hf."
            ) from exc

        self.model_id = model_id
        self.is_base = is_base
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._adapter_path = adapter_path

        enable_lora = adapter_path is not None
        self._llm = LLM(
            model=model_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=enable_lora,
            max_lora_rank=64,
        )
        self._lora_request = None
        if enable_lora:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, adapter_path)

    def _render(self, conversation: Conversation, add_generation_prompt: bool) -> str:
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        return _gemma_format(conversation, add_generation_prompt)

    def _sampling_params(self, temperature: float, max_new_tokens: int, n: int):
        from vllm import SamplingParams

        stop = ["<end_of_turn>"]
        return SamplingParams(
            n=n,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=stop,
        )

    def chat_batch(
        self,
        conversations: list[Conversation],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
    ) -> list[list[str]]:
        prompts = [self._render(c, add_generation_prompt=True) for c in conversations]
        sp = self._sampling_params(temperature, max_new_tokens, n)
        outs = self._llm.generate(prompts, sp, lora_request=self._lora_request)
        return [[o.text.strip() for o in req.outputs] for req in outs]

    def continue_from_prefill(
        self,
        conversation: Conversation,
        prefill: str,
        *,
        n: int,
        temperature: float,
        max_new_tokens: int,
    ) -> list[str]:
        prompt = self._render(conversation, add_generation_prompt=True) + prefill
        sp = self._sampling_params(temperature, max_new_tokens, n)
        outs = self._llm.generate([prompt], sp, lora_request=self._lora_request)
        return [o.text.strip() for o in outs[0].outputs]
