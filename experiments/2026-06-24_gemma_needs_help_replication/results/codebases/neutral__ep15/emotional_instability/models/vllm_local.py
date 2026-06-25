"""Optional vLLM backend for fast local Gemma sampling.

vLLM dramatically speeds up the Section 2 sweep (thousands of rollouts/model).
It supports prefill via raw-prompt completion: we render the chat template
ourselves and stop on the end-of-turn marker. Selected with ``GEN_BACKEND=vllm``.
"""
from __future__ import annotations

from .base import ChatClient, GenConfig, Message
from .hf_local import _TURN_END, _TURN_START


class VLLMClient(ChatClient):
    def __init__(self, spec) -> None:
        super().__init__(spec)
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.llm = LLM(model=spec.model_id, dtype="bfloat16",
                       trust_remote_code=True)
        self.is_base = spec.kind == "base"

    def _render(self, messages: list[Message], prefill: str = "") -> str:
        if self.is_base:
            parts = []
            for m in messages:
                if m["role"] == "system":
                    parts.append(m["content"])
                else:
                    parts.append(f"{_TURN_START}{m['role']}\n{m['content']}{_TURN_END}")
            parts.append(f"{_TURN_START}model\n{prefill}")
            return "\n".join(parts)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return text + prefill

    def _sample(self, prompt: str, cfg: GenConfig) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            stop=list(cfg.stop) if cfg.stop else [_TURN_END],
        )
        out = self.llm.generate([prompt], params)[0]
        return [o.text.strip() for o in out.outputs]

    def generate(self, messages: list[Message], cfg: GenConfig) -> list[str]:
        return self._sample(self._render(messages), cfg)

    def generate_with_prefill(self, messages, prefill, cfg):
        return self._sample(self._render(messages, prefill=prefill), cfg)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return self.tokenizer.decode(ids)
