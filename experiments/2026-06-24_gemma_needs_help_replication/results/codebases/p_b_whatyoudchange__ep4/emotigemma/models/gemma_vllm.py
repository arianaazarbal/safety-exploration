"""Gemma target-model backend (vLLM).

Handles three cases used across the paper:
  * instruct models (gemma-3-*-it): apply the chat template.
  * base model (gemma-3-27b-pt): no chat template; raw text continuation, used
    by the Section 3 prefill study.
  * finetuned variants: load a LoRA adapter on top of the instruct base
    (Section 4 SFT/DPO + layer ablations).

vLLM is chosen over plain transformers because Section 2 needs 4000 responses
per model at temperature 1; vLLM's n>1 sampling and continuous batching make
that tractable on a single 27B model.
"""
from __future__ import annotations

from ..config import ModelSpec
from .base import Message, SampleParams


class GemmaVLLMModel:
    def __init__(self, spec: ModelSpec, tensor_parallel_size: int = 1,
                 max_model_len: int = 8192, _engine=None, _tokenizer=None):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.spec = spec
        self.name = spec.name
        # Base models can be prefilled (continue arbitrary text); instruct can too
        # via continue_final_message, but the paper only prefills for the base/
        # instruct comparison, where we drive it through the same code path.
        self.supports_prefill = True

        # For a "+adapter" variant, spec.hf_id is the (instruct) base weights and
        # spec.adapter_path is the LoRA adapter loaded on top.
        model_id = spec.hf_id
        self._lora = None
        if spec.adapter_path:
            from vllm.lora.request import LoRARequest
            self._lora = LoRARequest("ft", 1, spec.adapter_path)

        self._engine = _engine or LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_lora=spec.adapter_path is not None,
            max_lora_rank=64,
        )
        self._tok = _tokenizer or AutoTokenizer.from_pretrained(model_id)

    def _render(self, messages: list[Message], prefill: str | None) -> str:
        """Render the prompt string sent to vLLM."""
        if self.spec.chat:
            # Instruct path: use the official Gemma chat template. If a prefill is
            # supplied, append it as the start of the assistant turn and ask the
            # template to continue it.
            text = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            if prefill:
                text += prefill
            return text
        # Base path: no chat template. We linearise the conversation into a plain
        # transcript so the base model "continues" consistently (paper Section 3).
        parts = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m["role"]]
            parts.append(f"{tag}: {m['content']}")
        parts.append("Assistant:")
        text = "\n".join(parts)
        if prefill:
            text += " " + prefill
        return text

    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        params: SampleParams | None = None,
        prefill: str | None = None,
    ) -> list[str]:
        from vllm import SamplingParams

        params = params or SampleParams()
        prompt = self._render(messages, prefill)
        sp = SamplingParams(
            n=n,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
        )
        kwargs = {"lora_request": self._lora} if self._lora else {}
        outputs = self._engine.generate([prompt], sp, **kwargs)
        # vLLM returns continuation text only (excluding the prompt/prefill), which
        # is exactly what the paper scores.
        return [o.text for o in outputs[0].outputs]
