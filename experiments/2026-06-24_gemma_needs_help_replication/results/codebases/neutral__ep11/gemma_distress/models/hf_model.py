"""Local HuggingFace inference for Gemma (instruct, base, and LoRA fine-tunes).

Handles three cases:
  * instruct models -> apply the Gemma chat template
  * base / pretrained models -> no chat template; we feed a plain concatenation
    and rely on prefilling (Section 3)
  * fine-tuned models -> load the base instruct weights + a PEFT LoRA adapter
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import RUNTIME, ModelSpec
from .base import Message, ModelClient


class HFModelClient(ModelClient):
    def __init__(self, spec: ModelSpec, adapter_path: str | None = None):
        super().__init__(spec.name)
        self.spec = spec
        self.adapter_path = adapter_path

        quant_kwargs = {}
        if RUNTIME.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # left padding is required for correct batched decoder-only generation
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **quant_kwargs,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------
    def _render(self, messages: list[Message], prefill: str | None) -> str:
        """Turn a message list into a single prompt string."""
        if self.spec.is_base:
            # Base models have no chat template. Present the conversation as
            # labelled plain text (Section 3 uses prefilling on top of this).
            parts = []
            for m in messages:
                tag = {"system": "System", "user": "User",
                       "assistant": "Assistant"}.get(m["role"], m["role"])
                parts.append(f"{tag}: {m['content']}")
            parts.append("Assistant:")
            text = "\n".join(parts)
            if prefill:
                text = text + " " + prefill
            return text

        # Instruct models: use the official chat template.
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if prefill:
            text = text + prefill
        return text

    @torch.no_grad()
    def generate_batch(
        self,
        batch_messages: list[list[Message]],
        *,
        temperature: float,
        max_new_tokens: int,
        prefills: list[str | None] | None = None,
    ) -> list[str]:
        prefills = prefills or [None] * len(batch_messages)
        prompts = [self._render(m, p) for m, p in zip(batch_messages, prefills)]
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        do_sample = temperature and temperature > 0
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_new_tokens: int,
        prefill: str | None = None,
    ) -> str:
        return self.generate_batch(
            [messages],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            prefills=[prefill],
        )[0]

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
