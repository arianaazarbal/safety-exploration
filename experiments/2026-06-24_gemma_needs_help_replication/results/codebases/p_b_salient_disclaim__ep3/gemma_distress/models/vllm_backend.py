"""vLLM backend for fast bulk generation of the Section-2 sweeps.

vLLM gives a large throughput win for the ~4000-response-per-model sweeps and
supports prefilled assistant turns via the tokenizer chat template + raw
`generate`. It does not expose hidden states, so the interpretability work
(Appendix I) uses the HF backend instead.
"""

from __future__ import annotations

from config import ModelSpec
from .base import ChatModel, Message
from .hf import _render_base_prompt


class VLLMChatModel(ChatModel):
    def __init__(self, spec: ModelSpec, *, adapter_path: str | None = None,
                 max_model_len: int = 16384, **llm_kwargs):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.spec = spec
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        enable_lora = adapter_path is not None
        self.llm = LLM(
            model=spec.hf_id,
            max_model_len=max_model_len,
            enable_lora=enable_lora,
            **llm_kwargs,
        )
        self._lora_request = None
        if enable_lora:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("adapter", 1, adapter_path)

    def _render(self, messages: list[Message], prefill: str | None) -> str:
        if self.spec.kind == "base":
            text = _render_base_prompt(messages)
            return f"{text} {prefill}" if prefill else text
        if prefill is not None:
            msgs = list(messages) + [{"role": "assistant", "content": prefill}]
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True,
            )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def generate(self, messages, *, max_new_tokens=None, temperature=None,
                 n=1, prefill=None):
        from vllm import SamplingParams
        from config import MAX_NEW_TOKENS, TEMPERATURE
        max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        temperature = TEMPERATURE if temperature is None else temperature

        prompt = self._render(messages, prefill)
        params = SamplingParams(
            n=n, temperature=temperature, top_p=1.0, max_tokens=max_new_tokens,
        )
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        out = self.llm.generate([prompt], params, **kwargs)
        return [o.text for o in out[0].outputs]

    def generate_batch(self, prompts_messages: list[tuple[list[Message], str | None]],
                       *, max_new_tokens=None, temperature=None, n=1):
        """Vectorised generation over many conversations (one vLLM batch)."""
        from vllm import SamplingParams
        from config import MAX_NEW_TOKENS, TEMPERATURE
        max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        temperature = TEMPERATURE if temperature is None else temperature

        prompts = [self._render(m, p) for m, p in prompts_messages]
        params = SamplingParams(
            n=n, temperature=temperature, top_p=1.0, max_tokens=max_new_tokens,
        )
        kwargs = {}
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        out = self.llm.generate(prompts, params, **kwargs)
        return [[o.text for o in r.outputs] for r in out]
