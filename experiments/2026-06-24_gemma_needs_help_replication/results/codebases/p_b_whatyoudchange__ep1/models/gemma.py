"""Open-weight Gemma client (local HuggingFace transformers inference).

Handles both instruct (`-it`) and base/pretrained (`-pt`) checkpoints, response
prefilling for Section 3, and loading LoRA adapters from Section 4.

Base models have no chat template, so for them `chat`/`prefill_continue` render a
plain `User:/Model:` transcript (documented choice — see DESIGN.md). Instruct
models use the official Gemma chat template with assistant-turn continuation.
"""

from __future__ import annotations

import threading

import torch

from config import MAX_NEW_TOKENS, TEMPERATURE
from .base import ChatModel, Message

# A lightweight plain-text transcript for base (pretrained) models, which have no
# chat template. Kept simple and consistent so base/instruct prefills line up.
_BASE_USER_TAG = "User: "
_BASE_MODEL_TAG = "Model: "


class GemmaModel(ChatModel):
    def __init__(self, hf_id: str, name: str, kind: str = "instruct",
                 adapter_path: str | None = None, device_map: str = "auto",
                 dtype: str = "bfloat16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.kind = kind
        self.supports_prefill = True
        self._lock = threading.Lock()   # HF generate is not thread-safe

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype), device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render_instruct(self, messages: list[Message], *,
                         add_generation_prompt: bool, prefill: str | None = None,
                         ) -> str:
        msgs = [dict(m) for m in messages]
        if prefill is not None:
            msgs.append({"role": "assistant", "content": prefill})
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                continue_final_message=True,
            )
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    def _render_base(self, messages: list[Message], prefill: str | None = None) -> str:
        lines = []
        for m in messages:
            tag = _BASE_USER_TAG if m["role"] in ("user", "system") else _BASE_MODEL_TAG
            lines.append(f"{tag}{m['content']}")
        # Open the model turn (and seed it with `prefill` if continuing one).
        opening = _BASE_MODEL_TAG + (prefill or "")
        lines.append(opening)
        return "\n".join(lines)

    def _render(self, messages: list[Message], *, prefill: str | None = None) -> str:
        if self.kind == "base":
            return self._render_base(messages, prefill=prefill)
        return self._render_instruct(
            messages, add_generation_prompt=(prefill is None), prefill=prefill)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate(self, prompt_text: str, *, n: int, max_new_tokens: int,
                  temperature: float) -> list[str]:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with self._lock, torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completions = []
        for seq in out:
            new_tokens = seq[prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(text.strip())
        return completions

    def chat(self, messages: list[Message], *, n: int = 1,
             max_new_tokens: int = MAX_NEW_TOKENS,
             temperature: float = TEMPERATURE) -> list[str]:
        prompt = self._render(messages)
        return self._generate(prompt, n=n, max_new_tokens=max_new_tokens,
                              temperature=temperature)

    def prefill_continue(self, messages: list[Message], prefill: str, *,
                         n: int = 1, max_new_tokens: int = MAX_NEW_TOKENS,
                         temperature: float = TEMPERATURE) -> list[str]:
        prompt = self._render(messages, prefill=prefill)
        return self._generate(prompt, n=n, max_new_tokens=max_new_tokens,
                              temperature=temperature)
