"""vLLM backend for high-throughput batched Gemma inference.

The elicitation sweep generates thousands of multi-turn rollouts per model, so
batched generation matters. vLLM's ``LLM.chat`` accepts a list of conversations
and applies the model's chat template. LoRA adapters (our DPO/SFT outputs) are
loaded via ``LoRARequest``.
"""
from __future__ import annotations

from ..config import SamplingConfig
from .base import ChatMessage, GenerationError, ModelClient


class VLLMClient(ModelClient):
    def __init__(
        self,
        model_id: str,
        spec_key: str,
        *,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
    ):
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.spec_key = spec_key
        self._adapter_path = adapter_path
        self._lora_request = (
            LoRARequest("adapter", 1, adapter_path) if adapter_path else None
        )
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_lora=adapter_path is not None,
            max_lora_rank=64,  # matches Appendix E LoRA rank
        )

    def _sampling_params(self, sampling: SamplingConfig):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
        )

    def _chat(self, conversations, sampling: SamplingConfig, *, continue_final: bool = False):
        params = self._sampling_params(sampling)
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        if continue_final:
            # Ask vLLM to continue the final (assistant) message rather than start
            # a new turn -- this implements prefill continuation.
            kwargs["add_generation_prompt"] = False
            kwargs["continue_final_message"] = True
        outputs = self.llm.chat(conversations, params, **kwargs)
        return [o.outputs[0].text for o in outputs]

    def generate(self, messages: list[ChatMessage], sampling: SamplingConfig) -> str:
        return self.generate_batch([messages], sampling)[0]

    def generate_batch(
        self, batch: list[list[ChatMessage]], sampling: SamplingConfig
    ) -> list[str]:
        try:
            conversations = [[m.as_dict() for m in conv] for conv in batch]
            return self._chat(conversations, sampling)
        except Exception as e:  # noqa: BLE001
            raise GenerationError(str(e)) from e

    # -- prefill continuation (used in batch form by the prefill experiment) --
    def continue_chat(
        self, messages: list[ChatMessage], prefill: str, sampling: SamplingConfig
    ) -> str:
        conv = [m.as_dict() for m in messages] + [{"role": "assistant", "content": prefill}]
        full = self._chat([conv], sampling, continue_final=True)[0]
        return full

    def continue_chat_batch(
        self,
        items: list[tuple[list[ChatMessage], str]],
        sampling: SamplingConfig,
    ) -> list[str]:
        conversations = [
            [m.as_dict() for m in msgs] + [{"role": "assistant", "content": prefill}]
            for msgs, prefill in items
        ]
        return self._chat(conversations, sampling, continue_final=True)

    # -- raw completion (base models loaded under vLLM, if desired) -----------
    def supports_completion(self) -> bool:
        return True

    def complete_batch(self, prompts: list[str], sampling: SamplingConfig) -> list[str]:
        params = self._sampling_params(sampling)
        kwargs = {"lora_request": self._lora_request} if self._lora_request else {}
        outputs = self.llm.generate(prompts, params, **kwargs)
        return [o.outputs[0].text for o in outputs]

    def complete(self, prompt_text: str, sampling: SamplingConfig) -> str:
        return self.complete_batch([prompt_text], sampling)[0]
