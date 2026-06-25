"""Gemma target-model client via HuggingFace transformers.

Handles three jobs the paper needs from open weights:

1. Instruct chat sampling (Section 2) using the Gemma chat template.
2. Response prefilling for both instruct and base checkpoints (Section 3): we
   render the prompt, append a prefill string that begins the assistant turn,
   generate, and return only the continuation.
3. Loading LoRA adapters produced by the Section 4 DPO/SFT training so the
   finetuned variants can be evaluated through the same interface.

The underlying ``model`` and ``tokenizer`` are exposed so the internal-emotion
probe (Appendix I) can read hidden states / logits.
"""

from __future__ import annotations

import torch

from .base import ChatMessage, ModelClient


class GemmaClient(ModelClient):
    supports_prefill = True

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        is_base: bool = False,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.name = name or model_id
        self.is_base = is_base

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {"device_map": device_map, "torch_dtype": getattr(torch, dtype)}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[ChatMessage], add_generation_prompt: bool) -> str:
        """Render the conversation to a single prompt string.

        Instruct models use the Gemma chat template. Base models are not chat
        tuned, so we use a minimal, explicit transcript format (CHOICE; see
        DESIGN.md) — this is only ever used together with a prefill in Section 3.
        """
        if self.is_base:
            lines = []
            for m in messages:
                tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
                lines.append(f"{tag}: {m.content}")
            if add_generation_prompt:
                lines.append("Assistant:")
            return "\n".join(lines)

        chat = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    @torch.no_grad()
    def _generate(self, prompt: str, temperature: float, max_new_tokens: int, n: int) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        completions = self.tokenizer.batch_decode(
            gen[:, prompt_len:], skip_special_tokens=True
        )
        return [c.strip() for c in completions]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        n: int = 1,
    ) -> list[str]:
        prompt = self._render(messages, add_generation_prompt=True)
        return self._generate(prompt, temperature, max_new_tokens, n)

    def continue_from(
        self,
        messages: list[ChatMessage],
        prefill: str,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        n: int = 1,
    ) -> list[str]:
        # Render up to (and including) the assistant generation prompt, then glue
        # the prefill onto it. The continuation excludes the prefill text.
        prompt = self._render(messages, add_generation_prompt=True) + prefill
        return self._generate(prompt, temperature, max_new_tokens, n)

    @torch.no_grad()
    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
