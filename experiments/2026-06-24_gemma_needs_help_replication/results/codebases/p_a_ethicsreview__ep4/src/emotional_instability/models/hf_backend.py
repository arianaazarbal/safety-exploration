"""Local HuggingFace backend for Gemma (instruct + base).

Supports the three things the API backends cannot:

* assistant *prefilling* -- continue a partial assistant turn. For instruct
  models we splice the prefill after the model role tag in the chat template;
  for base ("pt") models there is no chat template, so we treat the rendered
  conversation as plain text and continue it (Section 3 uses base models purely
  via prefilled continuations).
* tokenizer access for token-level truncation (Section 3 truncation points).
* hidden-state access for the Appendix I logit-based emotion probing.
"""

from __future__ import annotations

from typing import Optional

from .base import ChatModel, Message


class HFChatModel(ChatModel):
    def __init__(self, name: str, hf_id: str, role: str, dtype: str = "bfloat16",
                 device_map: str = "auto", adapter_path: str | None = None):
        self.name = name
        self.hf_id = hf_id
        self.role = role               # "instruct" | "base"
        self.supports_prefill = True
        self.supports_activations = True
        self._dtype = dtype
        self._device_map = device_map
        self._adapter_path = adapter_path   # optional LoRA adapter (DPO/SFT finetune)
        self._model = None
        self._tokenizer = None

    # -- lazy loading --------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self._dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=dtype, device_map=self._device_map
        )
        if self._adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
        self._model.eval()

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    # -- prompt construction -------------------------------------------------
    def build_input_text(
        self, messages: list[Message], assistant_prefill: Optional[str]
    ) -> str:
        """Render the conversation to the exact string fed to the model.

        Instruct models use the tokenizer's chat template with a generation
        prompt; an ``assistant_prefill`` is appended after that prompt so the
        model continues it. Base models have no chat template, so we render a
        plain transcript -- the prefill is what makes a base model produce a
        coherent assistant continuation at all (Section 3.1).
        """
        tok = self.tokenizer
        if self.role == "instruct" and tok.chat_template:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = self._render_plaintext(messages)
        if assistant_prefill:
            text = text + assistant_prefill
        return text

    @staticmethod
    def _render_plaintext(messages: list[Message]) -> str:
        lines = []
        for m in messages:
            lines.append(f"{m['role'].capitalize()}: {m['content']}")
        lines.append("Assistant:")
        return "\n".join(lines) + " "

    # -- generation ----------------------------------------------------------
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_new_tokens: int,
        seed: Optional[int] = None,
        assistant_prefill: Optional[str] = None,
    ) -> str:
        import torch

        self._ensure_loaded()
        if seed is not None:
            torch.manual_seed(seed)

        text = self.build_input_text(messages, assistant_prefill)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        prompt_len = inputs["input_ids"].shape[1]

        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id
                or self._tokenizer.eos_token_id,
            )
        # Return only newly generated tokens (the continuation), excluding any
        # prefill -- matching the Section 3 scoring convention.
        gen_ids = out[0][prompt_len:]
        return self._tokenizer.decode(gen_ids, skip_special_tokens=True)

    # -- helpers for the prefill experiment ---------------------------------
    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        """Return the first ``n_tokens`` tokens of ``text`` decoded back to a
        string -- used for the 'early' (20-token) truncation in Section 3.1."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
