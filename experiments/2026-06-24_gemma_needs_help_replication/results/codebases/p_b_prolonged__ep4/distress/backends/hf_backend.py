"""transformers backend for local Gemma inference.

Used for:
  * the base/pretrained Gemma models (no chat template) in the Section 3
    prefill experiment,
  * any case where vLLM is unavailable,
  * internal probing (Appendix I) needs raw hidden states, handled separately
    in distress.internal but reusing the model/tokenizer loaded here.

Prefill support is the key reason we keep a transformers path: continuing a
half-written assistant turn is awkward in serving frameworks but trivial here.
"""

from __future__ import annotations

from .base import ChatBackend, ChatMessage, GenResult
from ..config import GenConfig


class HFBackend(ChatBackend):
    supports_prefill = True

    def __init__(self, spec, device_map: str = "auto", dtype: str = "bfloat16", **kwargs):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        self.model.eval()
        self.is_instruct = spec.is_instruct

    # ------------------------------------------------------------------ #
    def _render(self, messages: list[ChatMessage], prefill: str | None = None) -> str:
        """Render a prompt string.

        Instruct models use the chat template. Base/pretrained models have no
        chat template, so we fall back to a plain transcript rendering (this is
        the regime the Section 3 prefill experiment is designed for: base models
        only ever *continue* a prefilled response, so formatting is light).
        """
        if self.is_instruct and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Plain transcript for base models.
            parts = []
            for m in messages:
                tag = {"system": "System", "user": "User", "assistant": "Assistant"}.get(m["role"], m["role"])
                parts.append(f"{tag}: {m['content']}")
            parts.append("Assistant:")
            text = "\n".join(parts)
        if prefill:
            text = text + prefill
        return text

    def _generate_from_text(self, text: str, gen: GenConfig, prefill_len_chars: int = 0) -> GenResult:
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = gen.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                do_sample=do_sample,
                temperature=gen.temperature if do_sample else None,
                top_p=gen.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][prompt_len:]
        completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenResult(
            text=completion,
            prompt_tokens=prompt_len,
            completion_tokens=int(new_tokens.shape[0]),
        )

    # ------------------------------------------------------------------ #
    def generate(self, messages: list[ChatMessage], gen: GenConfig) -> GenResult:
        text = self._render(messages)
        return self._generate_from_text(text, gen)

    def generate_prefill(self, messages, prefill, gen) -> GenResult:
        text = self._render(messages, prefill=prefill)
        # The model continues *after* `prefill`; decode returns only new tokens,
        # so the result already excludes the prefill.
        return self._generate_from_text(text, gen)
