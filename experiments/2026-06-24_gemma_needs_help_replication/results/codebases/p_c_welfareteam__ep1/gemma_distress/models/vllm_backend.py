"""vLLM backend for fast batched sampling of local Gemma models.

This is the recommended backend for the Section 2 evaluations (thousands of
rollouts at temperature 1).  It supports response prefilling via chat-template
continuation, but not residual-stream capture -- use the HF backend for
Appendix I.
"""
from __future__ import annotations

from ..config import ModelConfig
from .base import ChatModel, GenerationOptions, Message


class VLLMChatModel(ChatModel):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        self.llm = LLM(
            model=cfg.model_id,
            dtype=cfg.dtype,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            trust_remote_code=cfg.trust_remote_code,
            enable_lora=cfg.adapter_path is not None,
        )
        self._lora_request = None
        if cfg.adapter_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, cfg.adapter_path)
        self._has_chat_template = (
            not cfg.is_base_model and self.tokenizer.chat_template is not None
        )

    def _sampling_params(self, opts: GenerationOptions):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=opts.temperature,
            top_p=opts.top_p,
            max_tokens=opts.max_new_tokens,
            stop=opts.stop,
            seed=opts.seed,
        )

    def _generate_from_prompts(self, prompts: list[str], opts: GenerationOptions) -> list[str]:
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        outputs = self.llm.generate(prompts, self._sampling_params(opts), **kwargs)
        return [o.outputs[0].text for o in outputs]

    def _render(self, conversation: list[Message], prefill: str | None = None) -> str:
        if self._has_chat_template:
            if prefill:
                # Continue an assistant turn from a fixed prefix.
                conv = list(conversation) + [{"role": "assistant", "content": prefill}]
                text = self.tokenizer.apply_chat_template(
                    conv, tokenize=False, continue_final_message=True, add_generation_prompt=False
                )
                return text
            return self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        # Base model transcript (mirrors the HF backend rendering).
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in conversation]
        lines.append("Assistant:")
        text = "\n".join(lines)
        if prefill:
            text = text + " " + prefill
        return text

    def generate_batch(
        self, conversations: list[list[Message]], opts: GenerationOptions | None = None
    ) -> list[str]:
        o = self._resolved(opts)
        prompts = [self._render(c) for c in conversations]
        return self._generate_from_prompts(prompts, o)

    def supports_prefill(self) -> bool:
        return True

    def generate_with_prefill_batch(
        self,
        conversations: list[list[Message]],
        prefills: list[str],
        opts: GenerationOptions | None = None,
    ) -> list[str]:
        o = self._resolved(opts)
        prompts = [self._render(c, p) for c, p in zip(conversations, prefills)]
        return self._generate_from_prompts(prompts, o)
