"""Local HuggingFace backend for Gemma.

Used for:
  * Section 3 prefill study — base/pretrained Gemma (no chat template) must be
    given a prefilled assistant turn and asked to continue.
  * Section 4 — our LoRA-finetuned (DPO/SFT) adapters, which are not hosted on
    OpenRouter, must be run locally on top of the instruct base.

Loaded lazily so that importing the package on a machine without torch/GPU (to
run only the API-based Section 2 eval) does not fail.
"""
from __future__ import annotations

from ..config import HF_TOKEN, ModelSpec
from .base import ChatBackend, ChatMessage


class LocalHFBackend(ChatBackend):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if spec.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, spec.adapter_path)
            model = model.merge_and_unload()
        model.eval()
        self.model = model

    # --- prompt formatting -------------------------------------------------- #
    def _render_chat(self, messages: list[ChatMessage], add_generation_prompt: bool) -> str:
        """Apply Gemma's chat template for instruct models."""
        return self.tokenizer.apply_chat_template(
            [m.as_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _render_base(self, messages: list[ChatMessage]) -> str:
        """Plain concatenation for base models (no chat template). The prefill
        study supplies the assistant turn separately via continue_prefill, so
        here we just join prior turns as labelled text."""
        parts = []
        for m in messages:
            label = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
            parts.append(f"{label}: {m.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _generate_from_text(self, prompt_text: str, temperature: float, max_tokens: int) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=1.0,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    # --- ChatBackend API ---------------------------------------------------- #
    def generate(
        self,
        messages: list[ChatMessage],
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        if self.spec.is_base:
            text = self._render_base(messages)
        else:
            text = self._render_chat(messages, add_generation_prompt=True)
        return self._generate_from_text(text, temperature, max_tokens).strip()

    def continue_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        temperature: float = 1.0,
        max_tokens: int = 512,
    ) -> str:
        """Build the prompt up to (and including) the start of the assistant
        turn, append `prefill`, and generate. Returns only the continuation."""
        if self.spec.is_base:
            base_text = self._render_base(messages)  # ends with "Assistant:"
            prompt_text = f"{base_text} {prefill}"
        else:
            # Instruct: render with generation prompt, then append prefill INSIDE
            # the assistant turn (no closing turn token).
            prompt_text = self._render_chat(messages, add_generation_prompt=True) + prefill
        return self._generate_from_text(prompt_text, temperature, max_tokens)
