"""Local Gemma inference via vLLM.

vLLM is used for the open-weights targets (Gemma 3 27B / 12B, base and instruct)
because the Section-2 protocol samples thousands of responses at temperature 1
and vLLM's batched/continuous-batching throughput makes that tractable. The
same backend serves three needs:

  * instruct chat generation (apply the model's chat template),
  * base-model / prefill continuation (raw text, no template),
  * loading a LoRA adapter on top of the instruct base (Section 4 eval).

If a LoRA adapter path is supplied (via spec metadata or `adapter_path`), it is
attached with vLLM's LoRA support so the DPO/SFT models reuse the same code
path as vanilla Gemma.
"""

from __future__ import annotations

import os
from typing import Optional

from ..config import MAX_NEW_TOKENS, SAMPLING_TEMPERATURE, ModelSpec
from .base import Message, ModelBackend


class VLLMBackend(ModelBackend):
    def __init__(self, spec: ModelSpec, adapter_path: Optional[str] = None):
        super().__init__(spec)
        self.adapter_path = adapter_path or os.environ.get("EI_ADAPTER_PATH")
        self._llm = None
        self._tokenizer = None
        self._lora_request = None

    # -- lazy init so importing the module doesn't require a GPU ------------- #
    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        enable_lora = self.adapter_path is not None
        # Gemma-3-27B in bf16 needs sharding on most single nodes; default to
        # all visible GPUs. tensor_parallel_size is overridable via env.
        tp = int(os.environ.get("EI_TENSOR_PARALLEL", "1"))
        self._llm = LLM(
            model=self.spec.model_id,
            dtype="bfloat16",
            tensor_parallel_size=tp,
            enable_lora=enable_lora,
            max_lora_rank=64,            # matches training (Table 9)
            gpu_memory_utilization=float(os.environ.get("EI_GPU_UTIL", "0.90")),
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        if enable_lora:
            self._lora_request = LoRARequest("ei_adapter", 1, self.adapter_path)

    def _sampling_params(self, temperature: float, max_new_tokens: int, n: int):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=temperature,
            top_p=1.0,           # paper specifies temperature only; no nucleus trunc
            max_tokens=max_new_tokens,
            n=n,
        )

    # -- chat ---------------------------------------------------------------- #
    def chat(self, messages: list[Message], *,
             temperature: float = SAMPLING_TEMPERATURE,
             max_new_tokens: int = MAX_NEW_TOKENS,
             n: int = 1) -> list[str]:
        self._ensure_loaded()
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return self._generate(prompt, temperature, max_new_tokens, n)

    # -- raw continuation (base models / prefill) --------------------------- #
    def continue_text(self, prefix: str, *,
                      temperature: float = SAMPLING_TEMPERATURE,
                      max_new_tokens: int = MAX_NEW_TOKENS,
                      n: int = 1) -> list[str]:
        self._ensure_loaded()
        return self._generate(prefix, temperature, max_new_tokens, n)

    def _generate(self, prompt: str, temperature: float, max_new_tokens: int,
                  n: int) -> list[str]:
        params = self._sampling_params(temperature, max_new_tokens, n)
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outputs = self._llm.generate([prompt], params, **kwargs)
        return [o.text for o in outputs[0].outputs]

    def chat_prefix_prompt(self, messages: list[Message], prefill: str) -> str:
        """Render a chat prompt then append a `prefill` so the model continues
        from inside the assistant turn (used by the Section-3 prefill exp on the
        instruct model)."""
        self._ensure_loaded()
        base = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return base + prefill
