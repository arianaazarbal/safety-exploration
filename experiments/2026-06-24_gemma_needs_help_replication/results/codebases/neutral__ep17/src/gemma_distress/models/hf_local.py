"""Local HuggingFace/transformers backend.

Handles both chat (instruct) and raw-completion (base / "pt") Gemma models, and
supports loading a LoRA adapter on top of a base checkpoint (used to serve the
DPO / SFT finetunes from Section 4). Also supports prefilling the assistant
turn, which is essential for the Section 3 base-vs-instruct comparison.
"""
from __future__ import annotations

from typing import Any

from .base import ChatMessage, GenerationConfig, ModelClient


class HFClient(ModelClient):
    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        adapter = spec.get("adapter_path")
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    # ------------------------------------------------------------------
    def _render(self, messages: list[ChatMessage], prefill: str | None) -> str:
        if self.is_chat:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if prefill:
                text = text + prefill
            return text
        # Base model: no chat template. Concatenate as a transcript. The caller
        # (prefill experiment) is responsible for constructing a sensible
        # transcript; here we provide a minimal Role: content rendering.
        parts = []
        for m in messages:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append("Assistant:" + (f" {prefill}" if prefill else ""))
        return "\n".join(parts)

    def generate_n(self, messages: list[ChatMessage], cfg: GenerationConfig) -> list[str]:
        torch = self.torch
        text = self._render(messages, cfg.prefill)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        gen = self.model.generate(
            **inputs,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_tokens,
            num_return_sequences=cfg.n,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        outs = []
        for seq in gen:
            completion = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            # The prefill is part of the model "response" conceptually but the
            # judge scores only the *generated* continuation in the prefill
            # experiment, so we return only newly generated text here.
            outs.append(completion)
        return outs

    def complete(self, prompt: str, cfg: GenerationConfig) -> list[str]:
        torch = self.torch
        full = prompt + (cfg.prefill or "")
        inputs = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        gen = self.model.generate(
            **inputs,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_tokens,
            num_return_sequences=cfg.n,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        return [self.tokenizer.decode(s[prompt_len:], skip_special_tokens=True) for s in gen]
