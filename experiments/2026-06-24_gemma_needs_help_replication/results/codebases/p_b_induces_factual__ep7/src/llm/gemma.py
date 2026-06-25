"""Local Gemma inference via HuggingFace transformers, with assistant-prefill support.

This backend is used both as an evaluation *target* (Gemma 3 instruct/base) and as the
model being fine-tuned in Section 4. The same class loads a base checkpoint and, when
requested, attaches a trained LoRA adapter (the registry passes ``adapter_path``).

Notes on Gemma specifics:
- Gemma 3's chat template has no dedicated ``system`` role. We fold any system message
  into the first user turn (the conventional workaround), which matches how the calm /
  teacher system prompts are applied during data generation.
- Prefill (continuing a partially-written assistant turn) is implemented by rendering
  the chat template with ``add_generation_prompt=True`` and then appending the prefill
  text to the rendered string before tokenising — so the model literally continues from
  those tokens. We decode only the newly generated suffix.
"""
from __future__ import annotations

from typing import Optional

from .base import ChatModel, Message


class GemmaModel(ChatModel):
    supports_prefill = True

    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        adapter_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.name = name
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._dtype = dtype
        self._device_map = device_map
        self._load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

    # -- lazy loading -----------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_kwargs = {}
        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, self._dtype),
            device_map=self._device_map,
            **quant_kwargs,
        )
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    # -- message normalisation --------------------------------------------------
    @staticmethod
    def _fold_system(messages: list[Message]) -> list[Message]:
        """Gemma has no system role: prepend any system text to the first user turn."""
        sys_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if sys_text and rest and rest[0]["role"] == "user":
            rest = [{"role": "user", "content": f"{sys_text}\n\n{rest[0]['content']}"}] + rest[1:]
        elif sys_text:
            rest = [{"role": "user", "content": sys_text}] + rest
        return rest

    def _render(self, messages: list[Message], add_generation_prompt: bool) -> str:
        return self._tokenizer.apply_chat_template(
            self._fold_system(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _render_plain(messages: list[Message]) -> str:
        """Plain-text transcript rendering for *base* models (which are not chat-tuned).

        Used by the Section 3 prefill experiment: base models continue a plain
        "User:/Assistant:" transcript rather than the instruct chat template.
        """
        lines = []
        for m in messages:
            role = {"system": "System", "user": "User", "assistant": "Assistant"}[m["role"]]
            lines.append(f"{role}: {m['content']}")
        lines.append("Assistant:")  # generation prompt
        return "\n\n".join(lines) + " "

    # -- generation -------------------------------------------------------------
    def _generate_from_text(self, prompt_text: str, *, temperature: float, max_new_tokens: int) -> str:
        import torch

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._model.device)
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True)

    def generate(self, messages, *, temperature=1.0, max_new_tokens=1024, stop=None) -> str:
        self._ensure_loaded()
        prompt_text = self._render(messages, add_generation_prompt=True)
        text = self._generate_from_text(
            prompt_text, temperature=temperature, max_new_tokens=max_new_tokens
        )
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return text.strip()

    def generate_continuation(
        self, messages, prefill, *, temperature=1.0, max_new_tokens=512, chat_format=True
    ) -> str:
        self._ensure_loaded()
        base = self._render(messages, add_generation_prompt=True) if chat_format \
            else self._render_plain(messages)
        prompt_text = base + prefill
        cont = self._generate_from_text(
            prompt_text, temperature=temperature, max_new_tokens=max_new_tokens
        )
        return cont  # already excludes the prefill (decoded from after the prompt tokens)

    # -- tokenisation helpers (used by the prefill experiment) ------------------
    def count_tokens(self, text: str) -> int:
        self._ensure_loaded()
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def truncate_tokens(self, text: str, n_tokens: int, *, from_end: bool = False) -> str:
        """Return ``text`` truncated to the first/last ``n_tokens`` tokens."""
        self._ensure_loaded()
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        ids = ids[-n_tokens:] if from_end else ids[:n_tokens]
        return self._tokenizer.decode(ids, skip_special_tokens=True)
