"""Local HuggingFace transformers backend for Gemma (instruct + base).

Supports chat generation, raw-prompt continuation (for Section 3 prefill on base
models), and exposes the underlying model/tokenizer for internal-emotion probing
(Appendix I). Slower than vLLM; use vLLM for bulk Section 2 sampling.
"""
from __future__ import annotations

from typing import Any

from .base import Conversation, GenParams, ModelClient, ModelSpec


class HFModelClient(ModelClient):
    def __init__(self, spec: ModelSpec, adapter_path: str | None = None, **load_kwargs: Any):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        defaults = dict(torch_dtype=torch.bfloat16, device_map="auto")
        defaults.update(load_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **defaults)
        self.model.eval()
        self.adapter_path = adapter_path
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

    def _generate(self, input_ids, params: GenParams):
        import torch

        gen_kwargs = dict(
            max_new_tokens=params.max_new_tokens,
            do_sample=params.temperature > 0,
            temperature=max(params.temperature, 1e-5),
            top_p=params.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if params.seed is not None:
            torch.manual_seed(params.seed)
        out = []
        with torch.no_grad():
            for _ in range(params.n):
                gen = self.model.generate(input_ids, **gen_kwargs)
                # strip the prompt tokens
                cont = gen[0, input_ids.shape[1]:]
                out.append(self.tokenizer.decode(cont, skip_special_tokens=True))
        return out

    def generate_chat(self, conversation: Conversation, params: GenParams) -> list[str]:
        if not self.spec.chat:
            raise RuntimeError(
                f"{self.spec.name} is a base model; use continue_raw for prefill experiments."
            )
        messages = [{"role": m.role, "content": m.content} for m in conversation]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        return self._generate(input_ids, params)

    def continue_raw(self, prompt_text: str, params: GenParams) -> list[str]:
        """Continue from raw text without applying a chat template (base-model prefill)."""
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(
            self.model.device
        )
        return self._generate(input_ids, params)

    # --- helpers for internal probing (Appendix I) ---
    def build_prefill_text(self, conversation: Conversation, prefill: str) -> str:
        """Render a chat conversation and append an (unterminated) assistant prefill.

        Used to make a base model continue an assistant turn that starts with
        `prefill`. The instruct template's generation prompt is added, then the
        prefill text, so both base and instruct models continue from the same point.
        """
        messages = [{"role": m.role, "content": m.content} for m in conversation]
        rendered = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return rendered + prefill
