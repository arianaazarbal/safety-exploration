"""Local Gemma inference via transformers.

Supports the full set of operations the paper needs from an open-weight model:
standard chat, response prefilling/continuation (Section 3 + recovery probe),
optional LoRA-adapter loading (to evaluate the trained mitigation), and a hook
for the central-layer logit-based internal-emotion probe (Appendix I).
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ChatMessage, ModelClient


class GemmaClient(ModelClient):
    supports_prefill = True
    supports_logits = True

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.name = name or model_id
        self.model_id = model_id
        token = os.environ.get("HF_TOKEN")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        load_kwargs: dict = {"device_map": device_map, "token": token}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(
        self, messages: list[ChatMessage], *, prefill: str | None = None
    ) -> str:
        """Apply Gemma's chat template.

        Base (pre-trained) Gemma models have no instruct chat template, but
        transformers ships a default Gemma template for them too; the paper
        deliberately drives the base model with prefilled assistant turns so the
        exact wrapper matters little. If a `prefill` is given we open an
        assistant turn and continue it (`continue_final_message=True`).
        """
        has_template = getattr(self.tokenizer, "chat_template", None)
        if not has_template:
            # Base (pre-trained) Gemma has no chat template. Fall back to a plain
            # concatenation; the paper drives base models with prefills, so the
            # exact framing is secondary (documented in DESIGN.md).
            return self._render_plain(messages, prefill=prefill)

        if prefill is not None:
            msgs = list(messages) + [{"role": "assistant", "content": prefill}]
            return self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                continue_final_message=True,
            )
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _render_plain(messages: list[ChatMessage], *, prefill: str | None = None) -> str:
        parts = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
                m["role"], m["role"].capitalize()
            )
            parts.append(f"{tag}: {m['content']}")
        tail = "Assistant: " + (prefill or "")
        return "\n".join(parts + [tail])

    @torch.no_grad()
    def _generate(self, rendered: str, *, max_new_tokens: int, temperature: float):
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        token_strings = self.tokenizer.convert_ids_to_tokens(gen_ids)
        return text, token_strings

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(
        self, messages, *, max_new_tokens: int, temperature: float
    ) -> str:
        rendered = self._render(messages)
        text, _ = self._generate(
            rendered, max_new_tokens=max_new_tokens, temperature=temperature
        )
        return text.strip()

    def continue_from(
        self, messages, prefill: str, *, max_new_tokens: int, temperature: float
    ) -> str:
        rendered = self._render(messages, prefill=prefill)
        text, _ = self._generate(
            rendered, max_new_tokens=max_new_tokens, temperature=temperature
        )
        # `text` already excludes the prefill because it was part of the prompt.
        return text.strip()

    @torch.no_grad()
    def token_strings_for(
        self, messages, response: str
    ) -> list[str]:
        """Tokenize an existing assistant `response` (used for early-truncation
        at a fixed token count). Returns the token *strings* of the response."""
        ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        return self.tokenizer.convert_ids_to_tokens(ids)

    def detokenize(self, token_strings: list[str]) -> str:
        ids = self.tokenizer.convert_tokens_to_ids(token_strings)
        return self.tokenizer.decode(ids, skip_special_tokens=True)
