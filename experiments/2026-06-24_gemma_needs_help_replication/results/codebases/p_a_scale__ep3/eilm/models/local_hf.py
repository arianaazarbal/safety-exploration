"""Transformers backend for local Gemma models.

This is the fallback when vLLM is unavailable, and is also the backend used by
the internal-emotion probing (Appendix I), which needs hidden states / logits
that vLLM does not expose. Slower than vLLM; fine for the smaller probing runs.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .base import ChatClient, CompletionClient, GenConfig, GenResult, Message, fold_system

logger = logging.getLogger("eilm.hf")


class HFModel(ChatClient, CompletionClient):
    def __init__(
        self,
        hf_id: str,
        name: str,
        family: str = "gemma",
        role: str = "instruct",
        lora_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_id = hf_id
        self.name = name
        self.family = family
        self.role = role
        self._torch = torch

        self._tokenizer = AutoTokenizer.from_pretrained(hf_id)
        kwargs = dict(
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            trust_remote_code=True,
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        self._model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)

        if lora_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, lora_path)
        self._model.eval()

    def _gen_kwargs(self, cfg: GenConfig) -> dict:
        do_sample = cfg.temperature > 0
        return dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=do_sample,
            temperature=cfg.temperature if do_sample else None,
            top_p=cfg.top_p if do_sample else None,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        )

    def _decode_new(self, prompt_ids, output_ids) -> str:
        new = output_ids[prompt_ids.shape[-1]:]
        return self._tokenizer.decode(new, skip_special_tokens=True)

    def _run(self, text: str, cfg: GenConfig) -> GenResult:
        torch = self._torch
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, **{k: v for k, v in self._gen_kwargs(cfg).items() if v is not None})
        gen = self._decode_new(inputs["input_ids"][0], out[0])
        return GenResult(
            text=gen,
            usage={
                "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                "completion_tokens": int(out.shape[-1] - inputs["input_ids"].shape[-1]),
            },
        )

    def _render_chat(self, messages: List[Message]) -> str:
        if self.family == "gemma":
            messages = fold_system(messages)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # --- ChatClient --------------------------------------------------------
    def chat(self, messages: List[Message], cfg: GenConfig) -> GenResult:
        return self._run(self._render_chat(messages), cfg)

    # --- CompletionClient --------------------------------------------------
    def complete(self, prompt_text: str, cfg: GenConfig) -> GenResult:
        return self._run(prompt_text, cfg)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer
