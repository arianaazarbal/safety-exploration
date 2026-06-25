"""Optional vLLM backend for fast local Gemma sampling.

The paper samples 4000 responses per model at temperature 1; on a GPU, vLLM is
far faster than transformers' .generate for this volume. This backend mirrors
HFChatModel's interface. It is optional — if vLLM is not installed, the
registry falls back to the transformers backend.

Prefill continuation is supported via vLLM's `prompt` field (we render the
chat prompt and append the prefill, then strip it from the output).
"""
from __future__ import annotations

from typing import Any

from .base import ChatModel, GenerationParams, Message
from .hf_backend import HFChatModel


class VLLMChatModel(ChatModel):
    def __init__(
        self,
        name: str,
        hf_id: str,
        family: str,
        role: str,
        chat_template: str = "auto",
        adapter_path: str | None = None,
        tensor_parallel_size: int = 1,
    ):
        super().__init__(name=name, family=family, role=role)
        self.hf_id = hf_id
        self.chat_template = chat_template
        self.adapter_path = adapter_path
        self.tensor_parallel_size = tensor_parallel_size
        self._llm: Any = None
        self._tok: Any = None
        # Reuse HFChatModel's prompt-rendering logic (tokenizer-only; does not
        # load model weights).
        self._renderer = HFChatModel(
            name=name, hf_id=hf_id, family=family, role=role,
            chat_template=chat_template,
        )

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        kwargs: dict[str, Any] = dict(
            model=self.hf_id, tensor_parallel_size=self.tensor_parallel_size
        )
        if self.adapter_path:
            kwargs.update(enable_lora=True)
        self._llm = LLM(**kwargs)

    def _sampling_params(self, params: GenerationParams):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_new_tokens,
            stop=params.stop,
            seed=params.seed,
        )

    def _lora_request(self):
        if not self.adapter_path:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest("adapter", 1, self.adapter_path)

    def generate(self, messages: list[Message], params: GenerationParams) -> str:
        return self.generate_batch([messages], params)[0]

    def generate_batch(
        self, conversations: list[list[Message]], params: GenerationParams
    ) -> list[str]:
        self._ensure_loaded()
        prompts = [
            self._renderer._render(c, add_generation_prompt=True)
            for c in conversations
        ]
        outs = self._llm.generate(
            prompts, self._sampling_params(params), lora_request=self._lora_request()
        )
        return [o.outputs[0].text for o in outs]

    def continue_from_prefill(
        self, messages: list[Message], prefill: str, params: GenerationParams
    ) -> str:
        self._ensure_loaded()
        prompt = self._renderer._render(messages, add_generation_prompt=True) + prefill
        outs = self._llm.generate(
            [prompt], self._sampling_params(params), lora_request=self._lora_request()
        )
        return outs[0].outputs[0].text

    def count_tokens(self, text: str) -> int:
        return self._renderer.count_tokens(text)

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        return self._renderer.truncate_to_tokens(text, n_tokens)
