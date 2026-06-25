"""Local HuggingFace transformers backend for open-weight Gemma models.

This is the only backend that supports response *prefilling* (Section 3) and that
exposes hidden states / logits (Appendix I), because both require open weights.

Loading is lazy: importing this module does not require torch/transformers, so the
config/aggregation code can be used on a machine without the ML stack.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from ..config import ModelSpec
from .base import GenerationConfig, Message

if TYPE_CHECKING:  # pragma: no cover
    import torch


_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


class HFLocalModel:
    """Wraps a HuggingFace causal LM + tokenizer with chat and prefill helpers."""

    def __init__(self, spec: ModelSpec):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.name = spec.name
        self.is_base = spec.is_base_model
        torch_dtype = getattr(torch, _DTYPES.get(spec.dtype, "bfloat16"))

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        load_kwargs: dict = {"torch_dtype": torch_dtype, "device_map": "auto"}
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **load_kwargs)

        # Attach a LoRA adapter if this spec points at one (DPO/SFT outputs).
        if spec.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, spec.adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    def supports_prefill(self) -> bool:
        return True

    def _render_prompt(self, messages: list[Message], add_generation_prompt: bool) -> str:
        """Format messages into the model's prompt string.

        Base (pretrained) models have no chat template; we fall back to a plain
        role-tagged transcript so they can still be driven by prefilled text.
        """
        if self.is_base:
            parts = []
            for m in messages:
                parts.append(f"{m['role'].capitalize()}: {m['content']}")
            if add_generation_prompt:
                parts.append("Assistant:")
            return "\n\n".join(parts)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _generate(self, prompt_text: str, cfg: GenerationConfig) -> str:
        import torch

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-6),
            top_p=cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ #
    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        prompt = self._render_prompt(messages, add_generation_prompt=True)
        return self._generate(prompt, cfg)

    def continue_prefill(
        self, messages: list[Message], prefill: str, cfg: GenerationConfig
    ) -> str:
        """Generate a continuation of an assistant turn that begins with `prefill`.

        Returns only the *new* text (excludes the prefill), matching the paper's
        "score the continuation, excluding prefill" protocol.
        """
        prompt = self._render_prompt(messages, add_generation_prompt=True) + prefill
        return self._generate(prompt, cfg)

    # ------------------------------------------------------------------ #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        """Return the prefix of `text` that is the first `n_tokens` tokens."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)


@lru_cache(maxsize=2)
def _cached_model(name: str, spec_json: str) -> HFLocalModel:
    # spec_json keeps the lru_cache key hashable while still keying on full spec.
    spec = ModelSpec.model_validate_json(spec_json)
    return HFLocalModel(spec)


def load_hf_model(spec: ModelSpec) -> HFLocalModel:
    """Cached loader so repeated experiment calls reuse the in-memory 27B model."""
    return _cached_model(spec.name, spec.model_dump_json())
