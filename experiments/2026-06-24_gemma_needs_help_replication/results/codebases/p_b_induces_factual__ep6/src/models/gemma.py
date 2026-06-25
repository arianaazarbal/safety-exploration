"""Local HuggingFace inference for Gemma 3 (instruct + pretrained), with prefill
and optional LoRA-adapter loading for the finetuned variants.

Loaded lazily so that the rest of the codebase imports cleanly on machines without
GPUs / model weights. Heavy deps (torch, transformers, peft) are imported inside
``_ensure_loaded``.
"""

from __future__ import annotations

import config
from .base import ChatModel, Message


class GemmaLocalModel(ChatModel):
    supports_prefill = True

    def __init__(self, spec, *, adapter_path: str | None = None,
                 display_key: str | None = None, load_in_4bit: bool = False,
                 device_map: str = "auto"):
        self.spec = spec
        self.key = display_key or spec.key
        self.family = "gemma"
        self.is_base = spec.is_base
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_kwargs = {}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        # Left-pad for batched decoder generation; ensure a pad token exists.
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            **quant_kwargs,
        )
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    # ------------------------------------------------------------------ #
    def _render_prompt(self, messages: list[Message], prefill: str | None) -> str:
        """Render messages to a single prompt string.

        Instruct models use the chat template. Base (pretrained) models have no chat
        template, so we use a plain prefilled-continuation format (Section 3): the
        conversation is laid out as labelled turns and the model continues from the
        (possibly prefilled) assistant turn.
        """
        tok = self._tokenizer
        if not self.is_base and tok.chat_template:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            if prefill:
                text = text + prefill
            return text

        # Base-model fallback: simple role-labelled transcript.
        lines = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m["role"]]
            lines.append(f"{tag}: {m['content']}")
        lines.append("Assistant:" + (f" {prefill}" if prefill else " "))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def generate(self, messages, *, temperature=config.TEMPERATURE,
                 max_new_tokens=config.MAX_NEW_TOKENS, prefill=None) -> str:
        return self.generate_batch([messages], temperature=temperature,
                                   max_new_tokens=max_new_tokens,
                                   prefills=[prefill])[0]

    def generate_batch(self, batch, *, temperature=config.TEMPERATURE,
                       max_new_tokens=config.MAX_NEW_TOKENS, prefills=None) -> list[str]:
        self._ensure_loaded()
        import torch

        prefills = prefills or [None] * len(batch)
        prompts = [self._render_prompt(m, p) for m, p in zip(batch, prefills)]

        tok = self._tokenizer
        enc = tok(prompts, return_tensors="pt", padding=True).to(self._model.device)

        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )

        results = []
        input_len = enc["input_ids"].shape[1]
        for row in out:
            gen_ids = row[input_len:]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            results.append(text.strip())
        return results
