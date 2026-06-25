"""Optional vLLM backend for fast bulk generation.

Interchangeable with ``HFGemmaModel`` for the pure-generation experiments
(Section 2 elicitation, Section 4 evaluation), where the paper samples thousands
of responses per model. vLLM does not expose the residual stream, so probing
(Appendix I) must use the transformers backend.

Kept separate and import-light so the rest of the package does not depend on
vLLM being installed.
"""
from __future__ import annotations

from typing import Optional

from .base import GenerationConfig, ModelCapabilities, ModelInterface, Turn


class VLLMGemmaModel(ModelInterface):
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        is_base_model: bool = False,
        adapter_path: Optional[str] = None,
        max_model_len: int = 16384,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.name = name
        self.hf_id = hf_id
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        # LoRA adapters loaded via vLLM's LoRARequest at generation time.
        self._enable_lora = adapter_path is not None
        self._adapter_path = adapter_path
        self.llm = LLM(
            model=hf_id,
            enable_lora=self._enable_lora,
            max_lora_rank=64,
            max_model_len=max_model_len,
        )
        self.capabilities = ModelCapabilities(
            supports_internal_states=False,
            supports_prefill=True,
            is_base_model=is_base_model,
        )

    def _sampling_params(self, cfg: GenerationConfig):
        from vllm import SamplingParams

        return SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature,
            top_p=1.0,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )

    def _lora_request(self):
        if not self._enable_lora:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest("adapter", 1, self._adapter_path)

    def _render(self, messages: list[Turn], add_generation_prompt: bool) -> str:
        if self.capabilities.is_base_model:
            lines = [f"{t.role}: {t.content}" for t in messages]
            if add_generation_prompt:
                lines.append("assistant:")
            return "\n".join(lines)
        chat = [{"role": t.role, "content": t.content} for t in messages]
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def chat(self, messages: list[Turn], cfg: GenerationConfig) -> list[str]:
        prompt = self._render(messages, add_generation_prompt=True)
        outs = self.llm.generate(
            [prompt], self._sampling_params(cfg), lora_request=self._lora_request()
        )
        return [o.text for o in outs[0].outputs]

    def continue_from(
        self, messages: list[Turn], prefill: str, cfg: GenerationConfig
    ) -> list[str]:
        prompt = self._render(messages, add_generation_prompt=True) + prefill
        outs = self.llm.generate(
            [prompt], self._sampling_params(cfg), lora_request=self._lora_request()
        )
        return [o.text for o in outs[0].outputs]
