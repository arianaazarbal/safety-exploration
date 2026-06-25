"""Local Gemma 3 client (HuggingFace transformers).

Handles three things the rest of the pipeline needs:

1. Standard multi-turn chat generation via the Gemma chat template.
2. *Prefill* continuation: append a partially written assistant turn and let
   the model continue it (Section 3, Section 4 recovery test). Implemented by
   bypassing the closing turn token so generation resumes inside the assistant
   message.
3. Base-model continuation: ``google/gemma-3-*-pt`` has no chat template, so we
   fall back to raw text continuation of a prefill string (Section 3).

A loaded LoRA adapter (DPO/SFT output) can be attached via ``adapter_path``.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .base import ChatModel, Message

# Heavy imports are deferred to construction so that merely importing the
# package (e.g. for the API-only experiments) does not require torch.


class GemmaClient(ChatModel):
    supports_prefill = True

    def __init__(
        self,
        model_id: str,
        *,
        name: Optional[str] = None,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name or model_id
        self.model_id = model_id
        self.is_base = is_base

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            **quant_kwargs,
        )

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()
        self._torch = torch

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def _render_chat(self, messages: Sequence[Message], add_generation_prompt: bool) -> str:
        """Render messages with the Gemma chat template.

        Gemma 3's template has no dedicated system role; we fold a leading
        system message into the first user turn (the documented Gemma idiom).
        """
        msgs = self._fold_system(messages)
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _fold_system(messages: Sequence[Message]) -> list[Message]:
        msgs = list(messages)
        if msgs and msgs[0]["role"] == "system":
            sys = msgs[0]["content"]
            # Attach to the first user message.
            for i, m in enumerate(msgs[1:], start=1):
                if m["role"] == "user":
                    msgs[i] = {"role": "user", "content": f"{sys}\n\n{m['content']}"}
                    break
            msgs = msgs[1:]
        return msgs

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _generate_from_text(
        self, prompt_text: str, *, temperature: float, max_new_tokens: int, n: int
    ) -> list[str]:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[:, prompt_len:]
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in gen]

    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        n: int = 1,
    ) -> list[str]:
        if self.is_base:
            # Base models have no chat template; concatenate as plain text.
            text = self._render_base(messages)
        else:
            text = self._render_chat(messages, add_generation_prompt=True)
        return self._generate_from_text(
            text, temperature=temperature, max_new_tokens=max_new_tokens, n=n
        )

    def continue_prefill(
        self,
        messages: Sequence[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
        n: int = 1,
    ) -> list[str]:
        if self.is_base:
            text = self._render_base(messages) + prefill
        else:
            # Open the assistant turn, then inject the prefill so the model
            # continues *inside* the same assistant message (no closing token).
            text = self._render_chat(messages, add_generation_prompt=True) + prefill
        return self._generate_from_text(
            text, temperature=temperature, max_new_tokens=max_new_tokens, n=n
        )

    @staticmethod
    def _render_base(messages: Sequence[Message]) -> str:
        """Render a conversation as plain text for a base (pt) model.

        Section 3 only ever prefills base models with a single user task plus a
        prefilled assistant turn, so a light "User:/Assistant:" scaffold keeps
        the continuation coherent without imposing the instruct chat format.
        """
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(m["content"])
            elif m["role"] == "user":
                parts.append(f"User: {m['content']}")
            elif m["role"] == "assistant":
                parts.append(f"Assistant: {m['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts) + " "
