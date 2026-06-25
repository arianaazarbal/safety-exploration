"""vLLM backend -- the preferred local engine for the Gemma family.

Handles three things the experiments need:
  * chat for instruct models (Gemma chat template) and raw continuation for
    base/pretrained models;
  * prefill continuation, implemented by rendering the prompt up to the start of
    the assistant turn and concatenating the prefill text (so generation
    *continues* it);
  * optional LoRA adapter serving for the DPO / SFT finetunes (Section 4).

vLLM is imported lazily so that API-only experiments don't require it.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..config import GenConfig, DEFAULT_GEN, ModelSpec
from ..data_types import Conversation, to_openai
from .base import ModelClient, GenResult


class VLLMClient(ModelClient):
    supports_prefill = True

    def __init__(
        self,
        spec: ModelSpec,
        lora_path: Optional[str] = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
    ):
        from vllm import LLM  # lazy
        from vllm.lora.request import LoRARequest  # noqa: F401
        from transformers import AutoTokenizer

        self.spec = spec
        self.name = spec.name
        self.chat_templated = spec.chat_templated
        self._LoRARequest = LoRARequest

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.llm = LLM(
            model=spec.model_id,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=lora_path is not None,
            max_lora_rank=64,
        )
        self._lora_req = None
        if lora_path is not None:
            self._lora_req = LoRARequest(spec.name, 1, lora_path)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render_chat(self, messages: Conversation, add_generation_prompt: bool = True) -> str:
        """Render a conversation to a single prompt string.

        Instruct models use the Gemma chat template. Base models, which have no
        chat template, get a plain transcript -- but in practice base models are
        only ever driven through ``continue_prefill`` with a prefilled assistant
        turn, so this fallback is rarely used directly.
        """
        if self.chat_templated and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                to_openai(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Base-model fallback: minimal transcript.
        parts = []
        for m in messages:
            parts.append(f"{m.content}")
        return "\n\n".join(parts) + ("\n\n" if add_generation_prompt else "")

    def _sampling_params(self, gen: GenConfig):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_tokens,
            seed=gen.seed,
        )

    # ------------------------------------------------------------------ #
    # Chat
    # ------------------------------------------------------------------ #
    def chat(self, messages: Conversation, gen: GenConfig = DEFAULT_GEN) -> GenResult:
        return self.chat_batch([messages], gen)[0]

    def chat_batch(
        self, batch: Sequence[Conversation], gen: GenConfig = DEFAULT_GEN
    ) -> list[GenResult]:
        prompts = [self._render_chat(m, add_generation_prompt=True) for m in batch]
        outs = self.llm.generate(
            prompts, self._sampling_params(gen), lora_request=self._lora_req
        )
        return [GenResult(text=o.outputs[0].text) for o in outs]

    # ------------------------------------------------------------------ #
    # Prefill continuation
    # ------------------------------------------------------------------ #
    def continue_prefill(
        self, messages: Conversation, prefill: str, gen: GenConfig = DEFAULT_GEN
    ) -> GenResult:
        return self.continue_prefill_batch([(messages, prefill)], gen)[0]

    def continue_prefill_batch(
        self,
        batch: Sequence[tuple[Conversation, str]],
        gen: GenConfig = DEFAULT_GEN,
    ) -> list[GenResult]:
        prompts = []
        for messages, prefill in batch:
            base_prompt = self._render_chat(messages, add_generation_prompt=True)
            prompts.append(base_prompt + prefill)
        outs = self.llm.generate(
            prompts, self._sampling_params(gen), lora_request=self._lora_req
        )
        # vLLM returns only the newly generated text (the continuation).
        return [GenResult(text=o.outputs[0].text) for o in outs]

    # ------------------------------------------------------------------ #
    # Tokenisation helpers (truncation points for Section 3)
    # ------------------------------------------------------------------ #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return self.tokenizer.decode(ids)
