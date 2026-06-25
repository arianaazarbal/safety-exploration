"""Local HuggingFace backend for Gemma (instruct + base/pretrained).

Handles three things the rest of the pipeline relies on:
  1. Chat generation through Gemma's chat template (instruct models).
  2. Prefilled assistant continuations (Section 3) -- including for *base*
     models, which have no chat template, by formatting the conversation
     ourselves and letting the model continue raw text.
  3. Token counting / truncation for the "20 tokens into the turn" early-prefill.

LoRA adapters (from Section 4 training) are loaded via ``adapter_path``.
"""
from __future__ import annotations

from typing import Optional

import torch

from .base import Message, ModelBackend


class HFBackend(ModelBackend):
    def __init__(
        self,
        hf_id: str,
        name: Optional[str] = None,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name or hf_id
        self.hf_id = hf_id
        self.is_base = is_base
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)

        kwargs: dict = {"torch_dtype": getattr(torch, dtype), "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ---------------------------------------------------------------- helpers
    def _render(self, messages: list[Message], system: Optional[str], add_generation: bool) -> str:
        """Render the conversation to a prompt string.

        Instruct models use the chat template. Base models have none, so we use a
        plain transcript format (the paper compares base/instruct via prefilling,
        where exact formatting is shown not to matter much -- Appendix A.3).
        """
        if self.is_base:
            parts = []
            if system:
                parts.append(system)
            for m in messages:
                tag = "User" if m["role"] == "user" else "Assistant"
                parts.append(f"{tag}: {m['content']}")
            if add_generation:
                parts.append("Assistant:")
            return "\n\n".join(parts)

        chat = list(messages)
        if system:
            # Gemma has no system role; prepend system text to the first user turn.
            if chat and chat[0]["role"] == "user":
                chat = [{"role": "user", "content": f"{system}\n\n{chat[0]['content']}"}] + chat[1:]
            else:
                chat = [{"role": "user", "content": system}] + chat
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation
        )

    @torch.no_grad()
    def _sample(self, prompt: str, temperature, top_p, max_new_tokens, seed) -> str:
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature and temperature > 0
        out = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # ---------------------------------------------------------------- API
    def generate(self, messages, system=None, temperature=1.0, top_p=1.0,
                 max_new_tokens=2048, seed=None) -> str:
        prompt = self._render(messages, system, add_generation=True)
        return self._sample(prompt, temperature, top_p, max_new_tokens, seed)

    def continue_from(self, messages, prefill, temperature=1.0, top_p=1.0,
                      max_new_tokens=2048, seed=None) -> str:
        # Build prompt up to the assistant generation point, then append the
        # prefill text so the model continues it. We return only the new tokens.
        base_prompt = self._render(messages, None, add_generation=True)
        prompt = base_prompt + prefill
        return self._sample(prompt, temperature, top_p, max_new_tokens, seed)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
