"""Local Gemma client via HuggingFace transformers.

Handles three things the API clients cannot:

1. **Multi-turn generation** for instruct checkpoints via the chat template.
2. **Prefill continuation** (Section 3) — append a partial assistant string and
   let the model continue, returning only the new tokens. Works for both base
   (``-pt``) and instruct (``-it``) checkpoints.
3. **LoRA adapter loading** so the DPO/SFT finetuned variants (Section 4) can be
   evaluated with the same code path as the vanilla model.

Base (pretrained) checkpoints have no chat template, so for them we render the
conversation into a plain-text transcript (see ``_render_base_prompt``) before
prefilling — matching the paper's use of prefills to make base models "continue
the response" from the same starting points.
"""
from __future__ import annotations

from typing import Any

from ..config import ModelSpec
from .base import ChatClient, Message


class GemmaClient(ChatClient):
    supports_prefill = True

    def __init__(
        self,
        spec: ModelSpec,
        *,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.name = spec.name if adapter_path is None else f"{spec.name}+{adapter_path}"
        self.is_base = spec.kind == "base"
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(spec.ref)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": getattr(torch, dtype),
            "device_map": device_map,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(spec.ref, **model_kwargs)

        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()  # fold LoRA in for fast inference

        self.model.eval()

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------
    def _render_instruct_prompt(self, messages: list[Message], prefill: str = "") -> str:
        """Render messages with the Gemma chat template, leaving the generation
        prompt open. ``prefill`` (if any) is appended so the model continues it."""
        text = self.tokenizer.apply_chat_template(
            [dict(m) for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text + prefill

    def _render_base_prompt(self, messages: list[Message], prefill: str = "") -> str:
        """Base checkpoints have no chat format. Render a plain transcript so the
        model has a consistent starting point to continue from."""
        lines: list[str] = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}.get(
                m["role"], m["role"].capitalize()
            )
            lines.append(f"{tag}: {m['content']}")
        lines.append("Assistant: " + prefill)
        return "\n".join(lines)

    def _render(self, messages: list[Message], prefill: str = "") -> str:
        if self.is_base:
            return self._render_base_prompt(messages, prefill)
        return self._render_instruct_prompt(messages, prefill)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate_from_text(
        self, prompt_text: str, *, temperature: float, max_new_tokens: int, top_p: float
    ) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        return self._generate_from_text(
            self._render(messages),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )

    def prefill_continue(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> str:
        """Continue an assistant turn beginning with ``prefill``; return only the
        new continuation (the paper scores continuations excluding the prefill)."""
        return self._generate_from_text(
            self._render(messages, prefill),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
        )

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
