"""Local HuggingFace backend for Gemma (instruct, base, and LoRA-finetuned).

Handles three modes:
  * instruct chat       -- apply the Gemma chat template.
  * base completion     -- no chat template; concatenate turns as plain text
                           (used only in the Section 3 prefill experiment).
  * prefill continuation-- append `prefill` to the rendered prompt with the
                           assistant turn left open, then generate and return
                           only the newly generated text.

Optionally loads a PEFT/LoRA adapter on top of a base checkpoint so that the
DPO/SFT finetunes can be evaluated through the same code path.
"""
from __future__ import annotations

from typing import Any

import torch

from .base import ChatClient, Message


class HFLocalClient(ChatClient):
    supports_prefill = True

    def __init__(
        self,
        hf_id: str,
        *,
        kind: str = "instruct",
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        **_: Any,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_id = hf_id
        self.is_base = kind == "base"
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)

        load_kwargs: dict[str, Any] = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ---------------------------------------------------------------- prompts
    def _render_chat(self, messages: list[Message], add_generation_prompt: bool) -> str:
        """Render an instruct prompt via the Gemma chat template.

        Gemma's template has no system role; we fold any system message into the
        first user turn (the documented Gemma convention).
        """
        msgs = self._fold_system(messages)
        return self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in msgs],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _fold_system(messages: list[Message]) -> list[Message]:
        if messages and messages[0].role == "system":
            sys, rest = messages[0], messages[1:]
            if rest and rest[0].role == "user":
                merged = Message("user", f"{sys.content}\n\n{rest[0].content}")
                return [merged, *rest[1:]]
            return [Message("user", sys.content), *rest]
        return list(messages)

    def _render_base(self, messages: list[Message]) -> str:
        """Plain-text rendering for base models (no chat special tokens).

        Conversation is laid out as 'User: ...\nAssistant: ...' which base models
        continue naturally. Section 3 only ever uses this together with a prefill.
        """
        msgs = self._fold_system(messages)
        lines = []
        for m in msgs:
            label = "User" if m.role == "user" else "Assistant"
            lines.append(f"{label}: {m.content}")
        lines.append("Assistant:")
        return "\n".join(lines)

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def _generate(self, prompt_text: str, *, temperature, top_p, max_new_tokens, n, seed):
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = temperature > 0
        out = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen = out[:, prompt_len:]
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

    def chat(self, messages, *, temperature=1.0, top_p=1.0, max_new_tokens=2048, n=1, seed=None):
        if self.is_base:
            # Base models have no instruct format; expose completion of the
            # plain-text layout. (In practice Section 2 only scores instruct
            # models; base models go through continue_prefill.)
            prompt = self._render_base(messages)
        else:
            prompt = self._render_chat(messages, add_generation_prompt=True)
        return self._generate(
            prompt, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, n=n, seed=seed,
        )

    def continue_prefill(
        self, messages, prefill, *, temperature=1.0, top_p=1.0, max_new_tokens=2048, n=1, seed=None
    ):
        if self.is_base:
            prompt = self._render_base(messages) + " " + prefill
        else:
            # Open the assistant turn, then splice the prefill in before the model continues.
            prompt = self._render_chat(messages, add_generation_prompt=True) + prefill
        return self._generate(
            prompt, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, n=n, seed=seed,
        )
