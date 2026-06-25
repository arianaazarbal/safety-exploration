"""Local HuggingFace inference for Gemma (instruct + base), with optional LoRA
adapters and prefill/continuation support.

Handles three things the rest of the codebase relies on:

1. **Instruct vs base.** Instruct checkpoints (`-it`) use the tokenizer chat
   template; base/pretrained checkpoints (`-pt`) have no chat template, so we
   fall back to a plain-text rendering of the conversation. The Section 3
   experiment leans on this: base models only "continue" sensibly when prefilled.
2. **Prefill.** When ``prefill`` is given we make the model *continue* an
   assistant turn rather than start a fresh one, and return only the newly
   generated continuation (the paper scores "the generated continuation,
   excluding prefill").
3. **LoRA adapters.** Finetuned Gemma variants (Section 4) load the base
   instruct weights plus a PEFT adapter directory.

The underlying model/tokenizer are exposed via :attr:`model` / :attr:`tokenizer`
for the Appendix I logit-detection code, which needs hidden states and the
unembedding matrix.
"""

from __future__ import annotations

import os

from ..config import GENERATION, GenerationConfig, LORA_TARGET_MODULES, ModelSpec
from .base import GenResult, Message


class HFChatModel:
    """A Gemma chat/base model served locally via transformers."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        adapter_path: str | None = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.spec_key = spec.key
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {
            "torch_dtype": getattr(torch, torch_dtype),
            "device_map": device_map,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, torch_dtype),
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()  # fold adapter for fast inference

        self.model.eval()
        self._has_chat_template = self.tokenizer.chat_template is not None
        self.torch = torch

    # ------------------------------------------------------------------ #
    def supports_prefill(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], prefill: str | None) -> str:
        """Render the conversation to a single prompt string.

        For instruct models we use the chat template; with a prefill we keep the
        assistant turn open (``continue_final_message``). For base models we use
        a plain transcript, which is exactly the regime Section 3 targets.
        """
        msg_dicts = [m.to_dict() for m in messages]

        if self._has_chat_template:
            if prefill is not None:
                msg_dicts = msg_dicts + [{"role": "assistant", "content": prefill}]
                return self.tokenizer.apply_chat_template(
                    msg_dicts,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            return self.tokenizer.apply_chat_template(
                msg_dicts, tokenize=False, add_generation_prompt=True
            )

        # Base model: no chat template. Render a simple labelled transcript.
        lines: list[str] = []
        for m in messages:
            if m.role == "system":
                lines.append(m.content)
            elif m.role == "user":
                lines.append(f"User: {m.content}")
            else:
                lines.append(f"Assistant: {m.content}")
        text = "\n\n".join(lines) + "\n\nAssistant:"
        if prefill is not None:
            text = text + " " + prefill
        return text

    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: list[Message],
        *,
        gen: GenerationConfig = GENERATION,
        prefill: str | None = None,
    ) -> GenResult:
        torch = self.torch
        prompt = self._render(messages, prefill)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        # Per-call deterministic seeding so repeated samples differ but runs reproduce.
        if gen.seed is not None:
            torch.manual_seed(gen.seed)

        gen_kwargs: dict = dict(
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.temperature > 0,
            temperature=gen.temperature,
            top_p=gen.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if gen.top_k:
            gen_kwargs["top_k"] = gen.top_k

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenResult(
            text=text,
            prompt_tokens=int(prompt_len),
            completion_tokens=int(new_tokens.shape[0]),
            finish_reason="length" if new_tokens.shape[0] >= gen.max_new_tokens else "stop",
        )

    # ------------------------------------------------------------------ #
    def tokenize_count(self, text: str) -> int:
        """Number of tokens in ``text`` (used by token-based truncation)."""
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        """Return ``text`` truncated to its first ``n_tokens`` model tokens."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @staticmethod
    def lora_target_modules() -> list[str]:
        return list(LORA_TARGET_MODULES)
