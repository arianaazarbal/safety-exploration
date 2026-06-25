"""vLLM backend for fast batched Gemma sampling.

Used for the large temp=1 sweeps (Section 2) where we need thousands of
completions per model. Supports chat (via chat template) and raw completion.
Prefill of an assistant turn is done by rendering the chat template with
`continue_final_message=True` and feeding the raw string to the completion API.

Heavy imports are lazy.
"""
from __future__ import annotations

from .base import BaseClient, GenerationConfig, Message


class VLLMClient(BaseClient):
    def __init__(self, spec, gpu_memory_utilization: float = 0.90, tensor_parallel_size: int = 1):
        self.name = spec.name
        self.spec = spec
        self.is_base = spec.is_base
        self.supports_complete = True
        self._gpu_mem = gpu_memory_utilization
        self._tp = tensor_parallel_size
        self._llm = None
        self._tok = None

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest  # noqa: F401 (import validates support)

        self._tok = AutoTokenizer.from_pretrained(self.spec.model_id)
        kwargs = dict(
            model=self.spec.model_id,
            gpu_memory_utilization=self._gpu_mem,
            tensor_parallel_size=self._tp,
            max_model_len=self.spec.max_model_len,
        )
        if self.spec.adapter_path:
            kwargs["enable_lora"] = True
            self._lora = LoRARequest("adapter", 1, self.spec.adapter_path)
        else:
            self._lora = None
        self._llm = LLM(**kwargs)

    def _sampling_params(self, cfg: GenerationConfig):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop=cfg.stop,
            seed=cfg.seed,
        )

    # -- chat ----------------------------------------------------------------
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        return self.chat_batch([messages], cfg)[0]

    def chat_batch(self, batch, cfg):
        self._ensure_loaded()
        prompts = [self._render(m) for m in batch]
        return self._gen(prompts, cfg)

    def _render(self, messages, prefill: str | None = None) -> str:
        msgs = list(messages)
        add_gen = True
        if prefill is not None:
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            add_gen = False
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_gen,
            continue_final_message=prefill is not None,
        )

    def chat_with_prefill(self, messages, prefill, cfg):
        self._ensure_loaded()
        return self._gen([self._render(messages, prefill=prefill)], cfg)[0]

    # -- completion ----------------------------------------------------------
    def complete(self, prefix, cfg):
        return self.complete_batch([prefix], cfg)[0]

    def complete_batch(self, prefixes, cfg):
        self._ensure_loaded()
        return self._gen(list(prefixes), cfg)

    def _gen(self, prompts, cfg):
        params = self._sampling_params(cfg)
        kwargs = {}
        if self._lora is not None:
            kwargs["lora_request"] = self._lora
        outs = self._llm.generate(prompts, params, **kwargs)
        # vLLM preserves input order.
        return [o.outputs[0].text.strip() for o in outs]
