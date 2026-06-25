"""Local open-weight inference via HuggingFace transformers.

Used for the Gemma models (instruct + pretrained base). Supports:
  * standard multi-turn chat (instruct models, via the chat template),
  * raw prefill/continuation (base + instruct models, for Section 3),
  * loading a LoRA adapter on top of the base weights (for evaluating the
    DPO/SFT finetunes from Section 4).

For paper-scale runs of the 27B model, prefer the vLLM backend (vllm_model.py).
This transformers backend is correctness-first, not throughput-first.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ChatModel, GenerationResult


class HFChatModel(ChatModel):
    is_local = True

    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        is_base_model: bool = False,
    ):
        self.name = name
        self.hf_id = hf_id
        self.is_base_model = is_base_model
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        if adapter_path:
            # Lazy import so peft is only required when evaluating finetunes.
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    def _generate(self, input_ids, *, temperature: float, max_new_tokens: int) -> str:
        input_ids = input_ids.to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def chat(self, messages, *, temperature=1.0, max_new_tokens=2048) -> GenerationResult:
        if self.is_base_model:
            # Base models have no chat template; the prefill experiment is the
            # only sanctioned way to use them. Fall back to a plain concat.
            text = "\n".join(m["content"] for m in messages) + "\n"
            ids = self.tokenizer(text, return_tensors="pt").input_ids
        else:
            ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        return GenerationResult(self._generate(
            ids, temperature=temperature, max_new_tokens=max_new_tokens))

    def continue_assistant(self, messages, prefill, *, temperature=1.0,
                           max_new_tokens=2048) -> GenerationResult:
        """Render the conversation, open an assistant turn, append `prefill`,
        and let the model continue. We slice off the chat-template's trailing
        end-of-turn so generation flows directly from the prefill text."""
        if self.is_base_model:
            text = "\n".join(m["content"] for m in messages) + "\n" + prefill
        else:
            rendered = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            text = rendered + prefill
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        return GenerationResult(self._generate(
            ids, temperature=temperature, max_new_tokens=max_new_tokens))
