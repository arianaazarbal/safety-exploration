"""Local HuggingFace `transformers` backend.

Used wherever we need capabilities the API can't give us:
  * prefilled assistant turns + continuation scoring (Section 3),
  * raw access to the residual stream / unembedding (Appendix I),
  * inference on freshly-trained LoRA adapters (Section 4).

For the large Section-2 sweeps prefer the vLLM backend (much faster); this one
is correctness-first, not throughput-first.
"""

from __future__ import annotations

import torch

from config import ModelSpec
from .base import ChatModel, Message


def _render_base_prompt(messages: list[Message]) -> str:
    """Plain-text rendering of a conversation for *base* (pt) models.

    Base models were never trained on the Gemma chat template, so Section 3
    prefills them with a lightly-structured transcript and lets them continue.
    Appendix A.3 shows the exact chat formatting is not load-bearing.
    """
    parts: list[str] = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"])
        elif m["role"] == "user":
            parts.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"Assistant: {m['content']}")
    # Open the next assistant turn for continuation.
    if not messages or messages[-1]["role"] != "assistant":
        parts.append("Assistant:")
    return "\n\n".join(parts)


class HFChatModel(ChatModel):
    def __init__(self, spec: ModelSpec, *, adapter_path: str | None = None,
                 dtype: str = "bfloat16", device_map: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, torch_dtype=getattr(torch, dtype), device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- prompt construction ------------------------------------------------ #
    def _build_inputs(self, messages: list[Message], prefill: str | None):
        if self.spec.kind == "base":
            text = _render_base_prompt(messages)
            if prefill:
                # _render_base_prompt opens the turn with "Assistant:"; append
                # the prefill so the base model continues it.
                text = f"{text} {prefill}"
            return self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Instruct model: use the chat template.
        if prefill is not None:
            msgs = list(messages) + [{"role": "assistant", "content": prefill}]
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True,
            )
        else:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )

    # -- generation --------------------------------------------------------- #
    @torch.no_grad()
    def generate(self, messages, *, max_new_tokens=None, temperature=None,
                 n=1, prefill=None):
        from config import MAX_NEW_TOKENS, TEMPERATURE
        max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
        temperature = TEMPERATURE if temperature is None else temperature

        inputs = self._build_inputs(messages, prefill)
        prompt_len = inputs["input_ids"].shape[1]
        out = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen = out[:, prompt_len:]
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in gen]

    # -- interpretability hooks (Appendix I) -------------------------------- #
    @torch.no_grad()
    def residual_stream(self, messages, prefill=None):
        """Return (hidden_states, input_ids) for a single forward pass.

        hidden_states: tuple of (num_layers+1) tensors, each [1, seq, d_model].
        Used by internal/emotion_logits.py to unembed per-layer activations.
        """
        inputs = self._build_inputs(messages, prefill)
        out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states, inputs["input_ids"]

    @property
    def unembed(self) -> torch.nn.Module:
        """The output (lm_head) projection, used to map residual -> logits."""
        return self.model.get_output_embeddings()
