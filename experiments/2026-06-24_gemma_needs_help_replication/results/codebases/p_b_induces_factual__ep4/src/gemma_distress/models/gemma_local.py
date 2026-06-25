"""Local Gemma backend (HuggingFace Transformers).

This is the backend that supports everything the hosted API can't:

* **Base models** (``google/gemma-3-27b-pt``) for the Section 3 base-vs-instruct
  comparison — base models aren't chat-tuned, so we drive them purely through
  prefilled continuations.
* **Prefilling** — continue a partially written assistant turn (Section 3,
  Section 4.2 recovery study).
* **LoRA adapters** — load a DPO/SFT adapter on top of the instruct model to
  evaluate the finetuned variants (Section 4.2).

A single 27B model in bf16 needs ~54GB; pass ``load_in_4bit=True`` to fit on a
single 24-40GB GPU (qualitatively matches the paper, with minor numeric drift).
"""
from __future__ import annotations

from .base import ChatModel, Message, Role


class GemmaLocalModel(ChatModel):
    family = "gemma"

    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        is_base_model: bool = False,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.is_base_model = is_base_model
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)

        kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.name = f"{name}+{adapter_path.rstrip('/').split('/')[-1]}"
        self.model.eval()

    # ----------------------------------------------------------------- #
    def _render(self, messages: list[Message], prefill: str = "") -> str:
        """Render the conversation to a prompt string.

        For instruct models we use the chat template. For base models there is
        no template, so we fall back to a plain concatenation — the paper only
        ever uses base models via prefilled continuations, so the prompt body is
        effectively the prefill itself.
        """
        if self.is_base_model:
            # Base models: minimal scaffolding, rely on the prefill (Section 3).
            parts = [m.content for m in messages if m.role is not Role.SYSTEM]
            text = "\n\n".join(parts)
            return (text + "\n\n" + prefill) if prefill else text

        chat = [m.as_dict() for m in messages]
        prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt + prefill  # prefill = already-written start of the reply

    def _generate(self, prompt: str, *, temperature: float, max_tokens: int, n: int) -> list[str]:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]  # strip the prompt (incl. prefill)
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

    def chat(self, messages, *, temperature=1.0, max_tokens=1024) -> str:
        return self._generate(
            self._render(messages), temperature=temperature, max_tokens=max_tokens, n=1
        )[0]

    def continue_prefill(self, messages, prefill, *, n=1, temperature=1.0, max_tokens=1024):
        # Returns continuations only (prompt+prefill are stripped by _generate).
        return self._generate(
            self._render(messages, prefill),
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )
