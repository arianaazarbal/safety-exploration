"""Local HuggingFace transformers backend for Gemma (instruct + base).

Handles three things the experiments rely on:

* chat-templated multi-turn generation for instruct models;
* **prefilled** generation -- continue an assistant turn that we seed with a
  fixed string (Section 3). For instruct models we splice the prefill into the
  chat template *after* the assistant turn marker; for base ("pt") models,
  which have no chat template, we render the conversation as plain text and let
  the model continue from the prefill;
* loading a trained LoRA adapter (used to evaluate the DPO/SFT models).

The 27B model is large; ``load_in_4bit`` (bitsandbytes) is enabled by default so
it fits a single 48GB GPU. Override via the ``EI_LOAD_4BIT=0`` env var.
"""
from __future__ import annotations

import os

import torch

from .base import ChatClient, GenConfig, Message

_LOAD_4BIT = os.environ.get("EI_LOAD_4BIT", "1") == "1"

# Gemma 3 chat-template markers (used for base-model conversation rendering and
# for splicing prefills). The instruct template wraps turns in
# <start_of_turn>{role}\n ... <end_of_turn>.
_TURN_START = "<start_of_turn>"
_TURN_END = "<end_of_turn>"


class HFClient(ChatClient):
    def __init__(self, spec, adapter_path: str | None = None) -> None:
        super().__init__(spec)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        if _LOAD_4BIT:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.is_base = spec.kind == "base"

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], prefill: str = "") -> str:
        """Render ``messages`` to a prompt string, optionally ending mid-turn.

        For instruct models we use the tokenizer chat template with
        ``add_generation_prompt=True`` so the string ends right after the
        assistant turn marker, then append the prefill verbatim.

        For base models there is no chat template; we emulate the same surface
        form so prefilled continuations are comparable across base/instruct
        (Section 3.1's whole point).
        """
        if self.is_base:
            parts: list[str] = []
            for m in messages:
                if m["role"] == "system":
                    parts.append(m["content"])
                else:
                    parts.append(f"{_TURN_START}{m['role']}\n{m['content']}{_TURN_END}")
            parts.append(f"{_TURN_START}model\n{prefill}")
            return "\n".join(parts)

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text + prefill

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _sample(self, prompt: str, cfg: GenConfig) -> list[str]:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **enc,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=cfg.n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def generate(self, messages: list[Message], cfg: GenConfig) -> list[str]:
        # Base models cannot answer a bare chat prompt sensibly; require prefill.
        prompt = self._render(messages, prefill="")
        return self._sample(prompt, cfg)

    def generate_with_prefill(
        self, messages: list[Message], prefill: str, cfg: GenConfig
    ) -> list[str]:
        prompt = self._render(messages, prefill=prefill)
        return self._sample(prompt, cfg)

    # ------------------------------------------------------------------ #
    # Token helpers (used for the 20-token "early" truncation, etc.)
    # ------------------------------------------------------------------ #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:n_tokens]
        return self.tokenizer.decode(ids)
