"""Optional vLLM backend for high-throughput local Gemma sampling (Section 2).

vLLM is much faster than transformers for the 4000-sample sweeps, and supports
prefill continuation (just append the prefill text to the prompt and generate).
It does NOT expose hidden states, so the logit-lens probe (Appendix I) must use
`HFBackend`. Enable by setting `backend: vllm` on a model in the config.
"""

from __future__ import annotations

from .base import GenConfig, ModelBackend, Turn


class VLLMBackend(ModelBackend):
    supports_prefill = True
    supports_hidden_states = False

    def __init__(
        self,
        name: str,
        hf_id: str,
        family: str = "gemma",
        kind: str = "instruct",
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        adapter_path: str | None = None,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.name = name
        self.family = family
        self.kind = kind
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self._has_chat_template = self.tokenizer.chat_template is not None
        enable_lora = adapter_path is not None
        self.llm = LLM(
            model=hf_id,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_lora=enable_lora,
            dtype="bfloat16",
        )
        self._lora_request = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("adapter", 1, adapter_path)

    def _render_prompt(self, messages: list[Turn]) -> str:
        if self._has_chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        return "\n".join(lines) + "\nAssistant:"

    def _sample(self, prompt: str, gen: GenConfig) -> str:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=gen.temperature,
            top_p=gen.top_p,
            max_tokens=gen.max_new_tokens,
        )
        out = self.llm.generate([prompt], params, lora_request=self._lora_request)
        return out[0].outputs[0].text

    def chat(self, messages: list[Turn], gen: GenConfig | None = None) -> str:
        return self._sample(self._render_prompt(messages), gen or GenConfig()).strip()

    def prefill_continue(self, messages: list[Turn], prefill: str, gen: GenConfig | None = None) -> str:
        return self._sample(self._render_prompt(messages) + prefill, gen or GenConfig())

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
