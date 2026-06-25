"""Local Gemma inference backend (vLLM-backed).

Handles the instruct *and* base/pretrained Gemma checkpoints, and optionally a
LoRA adapter produced by the Section 4 training pipeline. vLLM is used because
the Section 2 sweeps need ~4000 temperature-1 samples per model and vLLM batches
those far more efficiently than vanilla ``transformers.generate``.

For the residual-stream probing in Appendix I we need hidden states, which vLLM
does not expose; that path uses ``transformers`` directly and lives in
``gemma_distress.probing`` rather than here.
"""
from __future__ import annotations

from typing import Any

from .base import ChatModel, Message


class HFLocalModel(ChatModel):
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        is_base: bool = False,
        adapter_path: str | None = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        # Imported lazily so the rest of the package (config, prompts, metrics,
        # API judges) is usable without a GPU / vLLM install.
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self.name = name
        self.is_base = is_base
        self._LoRARequest = LoRARequest
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)

        self.llm = LLM(
            model=hf_id,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=adapter_path is not None,
            max_lora_rank=64,
        )
        self._lora = (
            LoRARequest("intervention", 1, adapter_path)
            if adapter_path is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render_chat(self, conversation: list[Message], add_generation_prompt: bool) -> str:
        """Render a chat conversation to a prompt string.

        Base Gemma checkpoints have no chat template, so we fall back to a plain
        role-tagged transcript (matching how the prefill experiment treats base
        models — content matters more than chat format, per Appendix B Fig. 11).
        """
        if self.is_base:
            return self._render_plain(conversation, add_generation_prompt)
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _render_plain(conversation: list[Message], add_generation_prompt: bool) -> str:
        lines = []
        for m in conversation:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
            lines.append(f"{tag}: {m['content']}")
        text = "\n".join(lines)
        if add_generation_prompt:
            text += "\nAssistant:"
        return text

    def _render_continue(self, conversation: list[Message]) -> str:
        """Render a conversation whose final assistant turn is to be continued."""
        if conversation[-1]["role"] != "assistant":
            raise ValueError("continue_assistant requires a final assistant message")
        if self.is_base:
            # Base models: render preceding turns plain, then leave the prefill
            # dangling for the model to continue.
            return self._render_plain(conversation, add_generation_prompt=False)
        # Instruct models: HF chat templates support continuing the final
        # assistant message (no closing turn token, no new generation prompt).
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _sample(self, prompts: list[str], temperature: float, max_new_tokens: int, n: int):
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_new_tokens,
            n=n,
        )
        kwargs: dict[str, Any] = {}
        if self._lora is not None:
            kwargs["lora_request"] = self._lora
        return self.llm.generate(prompts, params, **kwargs)

    def generate(
        self,
        conversations: list[list[Message]],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[list[str]]:
        prompts = [self._render_chat(c, add_generation_prompt=True) for c in conversations]
        outputs = self._sample(prompts, temperature, max_new_tokens, n)
        return [[o.text for o in req.outputs] for req in outputs]

    def continue_assistant(
        self,
        conversations: list[list[Message]],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        n: int = 1,
    ) -> list[list[str]]:
        prompts = [self._render_continue(c) for c in conversations]
        outputs = self._sample(prompts, temperature, max_new_tokens, n)
        return [[o.text for o in req.outputs] for req in outputs]
