"""vLLM backend for high-throughput sampling of Gemma instruct weights.

Section 2 alone needs 4,000 rollouts per model, each a multi-turn conversation
re-sampled at temperature 1. vLLM's continuous batching makes this tractable on
a single node. We apply the model's own chat template via the HF tokenizer and
feed token-id prompts to the engine.

This backend is *not* used for prefill or probing (use ``HFLocalClient`` for
those); vLLM does not expose hidden states.
"""
from __future__ import annotations

from .base import ChatMessage, GenerationResult, SamplingParams


class VLLMClient:
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        adapter_path: str | None = None,
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
    ):
        # Imported lazily so the package is importable without a GPU/vLLM install
        # (e.g. for running the API-only or analysis paths, or the unit tests).
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.name = name
        self.hf_id = hf_id
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self._llm = LLM(
            model=hf_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=adapter_path is not None,
            max_lora_rank=64,
        )
        self._lora = (
            LoRARequest(name, 1, adapter_path) if adapter_path is not None else None
        )

    def _vllm_params(self, params: SamplingParams):
        from vllm import SamplingParams as VSP

        return VSP(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            stop=params.stop or None,
            seed=params.seed,
        )

    def _render(self, messages: list[ChatMessage]) -> str:
        return self.tokenizer.apply_chat_template(
            [m.as_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, messages: list[ChatMessage], params: SamplingParams) -> GenerationResult:
        return self.generate_batch([messages], params)[0]

    def generate_batch(
        self, conversations: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        prompts = [self._render(c) for c in conversations]
        outs = self._llm.generate(
            prompts, self._vllm_params(params), lora_request=self._lora
        )
        results = []
        for o in outs:
            comp = o.outputs[0]
            results.append(
                GenerationResult(
                    text=comp.text,
                    prompt_tokens=len(o.prompt_token_ids),
                    completion_tokens=len(comp.token_ids),
                    finish_reason=comp.finish_reason,
                    raw=o,
                )
            )
        return results
