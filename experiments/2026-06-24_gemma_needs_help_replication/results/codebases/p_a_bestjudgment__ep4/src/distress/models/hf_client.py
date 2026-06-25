"""Local generation via raw HuggingFace ``transformers``.

This backend is primarily for **base (pretrained) Gemma** in the Section 3 prefill
experiment, where we need precise control over tokenisation and prefilling and do
not need vLLM's throughput. It also doubles as a no-vLLM fallback for instruct
models and is the backend used by the internal-emotion probing (which needs hidden
states, so it instantiates the model directly — see ``probing/``).

Prefilling
----------
For base models we simply continue from a raw text prefix. For instruct models we
render the chat template up to the start of the assistant turn and append the
prefix, then decode only the newly generated tokens.
"""

from __future__ import annotations

from typing import Sequence

from .base import GenConfig, Message, ModelClient


class HFClient(ModelClient):
    supports_prefill = True

    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        is_chat: bool = False,
        device: str = "auto",
        dtype: str = "bfloat16",
        adapter_path: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.is_chat = is_chat
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
        )
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def _build_prompt_ids(self, messages: Sequence[Message], prefix: str = ""):
        if self.is_chat:
            text = self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        else:
            text = "".join(m["content"] for m in messages)
        text = text + prefix
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)

    def _generate_from_ids(self, inputs, cfg: GenConfig) -> str:
        torch = self.torch
        do_sample = cfg.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=cfg.top_p,
                max_new_tokens=cfg.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def generate(self, messages: Sequence[Message], cfg: GenConfig) -> str:
        return self._generate_from_ids(self._build_prompt_ids(messages), cfg)

    def prefill(self, messages: Sequence[Message], prefix: str, cfg: GenConfig) -> str:
        # Return only the continuation; the caller already holds ``prefix``.
        return self._generate_from_ids(self._build_prompt_ids(messages, prefix), cfg)
