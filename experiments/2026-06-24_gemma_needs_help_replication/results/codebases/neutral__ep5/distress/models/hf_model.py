"""Local HuggingFace inference for Gemma (instruct, base, and LoRA-adapted).

Handles:
  - instruct chat formatting via the Gemma chat template;
  - base-model evaluation through manual prompt construction + prefilling;
  - optional LoRA adapter loading for the finetuned variants (Section 4);
  - optional 4-bit quantisation so the 27B model fits on a single 24-48GB GPU.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ChatMessage, ModelClient

# Gemma turn markers (used for base-model formatting + prefill splicing).
GEMMA_BOS = "<bos>"
GEMMA_START = "<start_of_turn>"
GEMMA_END = "<end_of_turn>"


class HFChatModel(ModelClient):
    def __init__(
        self,
        key: str,
        model_id: str,
        *,
        is_base: bool = False,
        adapter_dir: str | None = None,
        load_in_4bit: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        self.key = key
        self.model_id = model_id
        self.is_base = is_base

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        quant_args = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device_map, **quant_args
        )
        if adapter_dir:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render_instruct(self, messages: list[ChatMessage]) -> str:
        """Render via the official chat template (Gemma folds system into user)."""
        msgs = self._fold_system(messages)
        return self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in msgs],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _render_base(self, messages: list[ChatMessage]) -> str:
        """Manual Gemma-style formatting for the base (pretrained) model.

        The base checkpoint isn't instruction-tuned, but the paper still uses
        the chat layout to keep base vs instruct comparable (Section 3.1).
        """
        msgs = self._fold_system(messages)
        parts = [GEMMA_BOS]
        for m in msgs:
            role = "model" if m.role == "assistant" else "user"
            parts.append(f"{GEMMA_START}{role}\n{m.content}{GEMMA_END}\n")
        parts.append(f"{GEMMA_START}model\n")
        return "".join(parts)

    @staticmethod
    def _fold_system(messages: list[ChatMessage]) -> list[ChatMessage]:
        """Gemma has no system role; prepend any system text to the first user turn."""
        sys_txt = "\n\n".join(m.content for m in messages if m.role == "system")
        rest = [m for m in messages if m.role != "system"]
        if sys_txt and rest and rest[0].role == "user":
            rest = [ChatMessage("user", f"{sys_txt}\n\n{rest[0].content}")] + rest[1:]
        return rest

    def _render(self, messages: list[ChatMessage]) -> str:
        return self._render_base(messages) if self.is_base else self._render_instruct(messages)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _generate(self, prompt_text: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = temperature > 0
        out = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return text.strip()

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048) -> str:
        return self._generate(self._render(messages), temperature, max_new_tokens)

    def chat_prefilled(self, messages, prefill, *, temperature=1.0, max_new_tokens=2048) -> str:
        # Append the prefill *inside* the open assistant turn, then continue.
        prompt_text = self._render(messages) + prefill
        return self._generate(prompt_text, temperature, max_new_tokens)

    # ------------------------------------------------------------------ #
    # Hooks for the internal-probing study (Appendix I)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def hidden_states(self, text: str):
        """Return per-layer hidden states for ``text`` (tuple of [seq, d_model])."""
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states, inputs["input_ids"][0]
