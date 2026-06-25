"""Local vLLM backend for high-throughput Gemma sampling (Section 2 bulk eval).

vLLM is optional. Construction raises a clear error if vllm is not installed, so the
rest of the package still imports. Supports batched chat and raw continuation.
"""
from __future__ import annotations

from typing import Any

from .base import Conversation, GenParams, ModelClient, ModelSpec


class VLLMModelClient(ModelClient):
    def __init__(self, spec: ModelSpec, adapter_path: str | None = None, **engine_kwargs: Any):
        super().__init__(spec)
        try:
            from vllm import LLM
            from vllm.lora.request import LoRARequest  # noqa: F401
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "vLLM backend requested but vllm is not importable. Install vllm, or set "
                f"backend: hf for {spec.name} in config/models.yaml."
            ) from e

        kwargs: dict[str, Any] = dict(model=spec.hf_id, dtype="bfloat16")
        if spec.max_model_len:
            kwargs["max_model_len"] = spec.max_model_len
        if adapter_path:
            kwargs["enable_lora"] = True
        kwargs.update(engine_kwargs)
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.adapter_path = adapter_path
        self._lora_request = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, adapter_path)

    def _sampling_params(self, params: GenParams):
        from vllm import SamplingParams

        return SamplingParams(
            n=params.n,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_new_tokens,
            stop=params.stop or None,
            seed=params.seed,
        )

    def _gen(self, prompts: list[str], params: GenParams) -> list[list[str]]:
        sp = self._sampling_params(params)
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outputs = self.llm.generate(prompts, sp, **kwargs)
        return [[o.text for o in out.outputs] for out in outputs]

    def _render_chat(self, conversation: Conversation) -> str:
        messages = [{"role": m.role, "content": m.content} for m in conversation]
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def generate_chat(self, conversation: Conversation, params: GenParams) -> list[str]:
        return self._gen([self._render_chat(conversation)], params)[0]

    def generate_chat_batch(
        self, conversations: list[Conversation], params: GenParams
    ) -> list[list[str]]:
        return self._gen([self._render_chat(c) for c in conversations], params)

    def continue_raw(self, prompt_text: str, params: GenParams) -> list[str]:
        return self._gen([prompt_text], params)[0]

    def continue_raw_batch(
        self, prompt_texts: list[str], params: GenParams
    ) -> list[list[str]]:
        return self._gen(prompt_texts, params)
