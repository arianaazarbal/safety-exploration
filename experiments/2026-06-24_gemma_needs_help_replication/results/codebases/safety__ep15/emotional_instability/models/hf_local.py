"""Local HuggingFace backend for Gemma (instruct + base/pretrained).

Handles three things the API backends do not:
  * loading a (possibly LoRA-adapted) Gemma checkpoint,
  * true assistant-turn **prefilling** for Section 3.1,
  * applying the Gemma chat template (instruct) or raw continuation (base).

Lazy imports keep ``torch``/``transformers`` off the critical path for
API-only runs.
"""
from __future__ import annotations

from typing import Sequence

from .base import ChatModel, Message


class HFLocalModel(ChatModel):
    def __init__(
        self,
        spec,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        adapter_path: str | None = None,
    ) -> None:
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..config import API_KEYS

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, token=API_KEYS.hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            token=API_KEYS.hf_token,
            **quant_kwargs,
        )
        if adapter_path is not None:
            # Attach a trained LoRA adapter (e.g. the DPO finetune).
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch

    # ------------------------------------------------------------------ #
    def _render(self, messages: Sequence[Message], *, add_generation_prompt: bool) -> str:
        """Render messages to a prompt string.

        Instruct models use the chat template. Base/pretrained models have no
        chat template, so we fall back to a plain transcript format (matching
        the paper's observation in Appendix A.3 that *content*, not chat
        formatting, drives the behaviour).
        """
        if self.spec.is_base:
            return self._render_plain(messages, add_generation_prompt)
        return self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _render_plain(messages: Sequence[Message], add_generation_prompt: bool) -> str:
        lines = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
            lines.append(f"{tag}: {m['content']}")
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines)

    def _generate(self, prompt_text: str, *, temperature, top_p, max_new_tokens) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        # Return only the newly generated tokens.
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ #
    def chat(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048) -> str:
        prompt_text = self._render(messages, add_generation_prompt=True)
        return self._generate(
            prompt_text, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)

    def continue_prefill(self, messages, prefill, *, temperature=1.0, top_p=1.0,
                         max_new_tokens=2048) -> str:
        # Render up to the generation prompt, then append the prefill text so the
        # model continues *from inside* the assistant turn.
        prompt_text = self._render(messages, add_generation_prompt=True) + prefill
        return self._generate(
            prompt_text, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens)
