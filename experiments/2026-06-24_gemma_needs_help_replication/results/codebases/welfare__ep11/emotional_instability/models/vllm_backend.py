"""vLLM backend for fast batched generation of local Gemma instruct models.

Section 2 samples 4000 responses per model at temperature 1; vLLM makes this
tractable. We use the chat API for instruct models and support prefilled
continuation via the tokenizer's chat template + `continue_final_message`.

If vLLM is unavailable, callers can force the HF backend with the
``--backend hf`` override (see config.RunConfig.backend_override).
"""

from __future__ import annotations

import os

from ..config import CHECKPOINTS_DIR, ModelSpec
from .base import Backend, Message


class VLLMBackend(Backend):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from vllm import LLM
        from transformers import AutoTokenizer

        enable_lora = spec.is_finetune
        self.llm = LLM(
            model=spec.hf_id,
            dtype="bfloat16",
            enable_lora=enable_lora,
            max_lora_rank=64,            # matches our LoRA rank (config.DPO/SFT)
            tensor_parallel_size=int(os.environ.get("EI_TP_SIZE", "1")),
            gpu_memory_utilization=float(os.environ.get("EI_GPU_MEM_UTIL", "0.90")),
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)

        self._lora_request = None
        if enable_lora:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest(spec.key, 1, str(CHECKPOINTS_DIR / spec.key))

    def _sampling_params(self, n, max_new_tokens, temperature, top_p):
        from vllm import SamplingParams

        return SamplingParams(
            n=n, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
            seed=None,  # let vLLM vary samples
        )

    def _render(self, messages: list[Message], prefill: str | None = None) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=prefill is None)
        if prefill is not None:
            # Append the prefill as the start of the assistant turn and keep it open.
            messages = list(messages) + [{"role": "assistant", "content": prefill}]
            kwargs = dict(tokenize=False, continue_final_message=True,
                          add_generation_prompt=False)
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, messages, n=1, max_new_tokens=2048, temperature=1.0, top_p=1.0):
        prompt = self._render(messages)
        params = self._sampling_params(n, max_new_tokens, temperature, top_p)
        kw = {}
        if self._lora_request is not None:
            kw["lora_request"] = self._lora_request
        outputs = self.llm.generate([prompt], params, **kw)
        return [o.text for o in outputs[0].outputs]

    def generate_batch(self, batch, max_new_tokens=2048, temperature=1.0, top_p=1.0):
        prompts = [self._render(m) for m in batch]
        params = self._sampling_params(1, max_new_tokens, temperature, top_p)
        kw = {}
        if self._lora_request is not None:
            kw["lora_request"] = self._lora_request
        outputs = self.llm.generate(prompts, params, **kw)
        # vLLM preserves input order.
        return [o.outputs[0].text for o in outputs]

    def supports_prefill(self) -> bool:
        return True

    def generate_with_prefill(self, messages, prefill, n=1, max_new_tokens=2048,
                              temperature=1.0, top_p=1.0):
        prompt = self._render(messages, prefill=prefill)
        params = self._sampling_params(n, max_new_tokens, temperature, top_p)
        kw = {}
        if self._lora_request is not None:
            kw["lora_request"] = self._lora_request
        outputs = self.llm.generate([prompt], params, **kw)
        # vLLM returns only the generated continuation (prompt incl. prefill is
        # the prefix), matching the paper's "exclude prefill" scoring.
        return [o.text for o in outputs[0].outputs]
