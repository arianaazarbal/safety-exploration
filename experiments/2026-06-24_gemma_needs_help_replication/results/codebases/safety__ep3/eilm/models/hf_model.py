"""Local HuggingFace backend for Gemma (instruct, base, and finetuned adapters).

This is the workhorse backend: it supports plain chat, prefilling/continuation
(Section 3), token-level truncation, and exposes the underlying model so the
internal-emotion probe (Appendix I) can read the residual stream.

Design notes (see DESIGN.md):
* Gemma-3 chat formatting is delegated to the tokenizer's chat template, so we
  never hand-format ``<start_of_turn>`` markers.
* For *base* models there is no chat template, so we build a plain-text
  transcript ourselves (``_format_base_transcript``) and always operate in
  continuation mode — this is exactly the regime Section 3 needs.
* A LoRA adapter directory can be supplied to load a finetuned model on top of
  the same base weights without re-downloading them.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ChatModel, Message


class HFModel(ChatModel):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        is_base: bool = False,
        adapter_path: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.name = name
        self.model_id = model_id
        self.is_base = is_base

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"torch_dtype": dtype, "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _format_base_transcript(self, messages: list[Message]) -> str:
        """Plain-text rendering for base models (no chat template).

        We use a light ``User:``/``Assistant:`` scaffold. The transcript ends
        with ``Assistant:`` so the model continues an assistant turn — the
        prefill regime from Section 3.1.
        """
        parts: list[str] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                parts.append(m["content"])
            elif role == "user":
                parts.append(f"User: {m['content']}")
            elif role == "assistant":
                parts.append(f"Assistant: {m['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _build_inputs(
        self, messages: list[Message], prefill: str | None = None
    ) -> str:
        """Return the full prompt string (pre-tokenisation)."""
        if self.is_base:
            text = self._format_base_transcript(messages)
            if prefill:
                text = text + " " + prefill
            return text

        # Instruct: use the chat template and open an assistant turn.
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            text = text + prefill
        return text

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _generate_from_text(
        self, prompt_text: str, temperature: float, max_new_tokens: int
    ) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(
            self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def generate(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        return self._generate_from_text(
            self._build_inputs(messages), temperature, max_new_tokens)

    @torch.no_grad()
    def generate_batch(
        self,
        batch: list[list[Message]],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> list[str]:
        prompts = [self._build_inputs(m) for m in batch]
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True).to(self.model.device)
        out = self.model.generate(
            **enc,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip()
                for g in gen]

    # ------------------------------------------------------------------ #
    # Prefilling
    # ------------------------------------------------------------------ #
    def supports_prefill(self) -> bool:
        return True

    def continue_from_prefill(
        self,
        messages: list[Message],
        prefill: str,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        prompt = self._build_inputs(messages, prefill=prefill)
        return self._generate_from_text(prompt, temperature, max_new_tokens)

    # ------------------------------------------------------------------ #
    # Tokenisation helpers
    # ------------------------------------------------------------------ #
    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)

    def n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
