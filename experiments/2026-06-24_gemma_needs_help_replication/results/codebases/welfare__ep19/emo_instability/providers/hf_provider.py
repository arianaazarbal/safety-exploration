"""Local HuggingFace/transformers backend for open-weight Gemma targets.

This is the provider the paper uses for Gemma ("for local inference we use the
following HuggingFace model identifiers ... google/gemma-3-27b-it,
google/gemma-3-27b-pt ..."). It is also the only backend that supports true
assistant-turn prefill / base-model continuation, which Sections 3 and 4.2 need.

Loading a 27B model needs a capable GPU (or multi-GPU via device_map=auto).
The eval suite can run entirely against API providers if local weights are
unavailable; this backend is required only for the Gemma experiments.
"""
from __future__ import annotations

import torch

from .base import ChatModel, GenConfig, Message

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HFModel(ChatModel):
    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        adapter_path: str | None = None,
        is_base: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.name = name or model_id
        self.is_base = is_base
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_DTYPES.get(dtype, torch.bfloat16),
            device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def supports_prefill(self) -> bool:
        return True

    # -- prompt construction -------------------------------------------------
    def _render(self, messages: list[Message], prefill: str | None) -> str:
        """Render messages to a raw prompt string.

        Instruct models use the chat template. Base ("pt") models are not
        chat-trained, so per the paper we feed a lightly-formatted transcript
        and rely on prefill to anchor the continuation.
        """
        if self.is_base:
            return self._render_base(messages, prefill)

        # Instruct: use the model's chat template, then append prefill (if any)
        # by continuing the final assistant turn.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            text = text + prefill
        return text

    @staticmethod
    def _render_base(messages: list[Message], prefill: str | None) -> str:
        lines = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
                m["role"], m["role"].title()
            )
            lines.append(f"{tag}: {m['content']}")
        lines.append("Assistant:")
        text = "\n".join(lines)
        if prefill:
            text = text + " " + prefill
        return text

    # -- generation ----------------------------------------------------------
    @torch.no_grad()
    def generate(
        self, messages: list[Message], cfg: GenConfig, prefill: str | None = None
    ) -> str:
        prompt = self._render(messages, prefill)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        do_sample = cfg.temperature > 0
        out = self.model.generate(
            **inputs,
            max_new_tokens=cfg.max_tokens,
            do_sample=do_sample,
            temperature=cfg.temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        # Return only the continuation (prefill is already in the prompt, not the
        # generated ids), matching the paper's "continuation excluding prefill".
        return text.strip()
