"""Local HuggingFace backend for Gemma models (instruct, base, and LoRA finetunes).

Supports:
  * chat-formatted generation for instruct models (Gemma-3 chat template);
  * raw prefill continuation for base models and the §3 study;
  * loading a PEFT LoRA adapter on top of the base weights (our DPO/SFT models);
  * exposing the underlying model/tokenizer for Appendix-I logit probing.

The 27B model is large; by default we load in bf16 and let ``accelerate`` place
it (``device_map="auto"``). 4-bit loading via bitsandbytes is available through
``load_in_4bit=True`` for single-GPU runs. We do NOT depend on vLLM here so the
same object can be reused for probing/finetuning; a vLLM fast path can be slotted
in for large sampling jobs (see DESIGN.md).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from gnh.config import MAX_NEW_TOKENS, TEMPERATURE, ModelSpec
from gnh.models.base import Message

if TYPE_CHECKING:  # heavy imports deferred to runtime
    import torch


class HFBackend:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.adapter_path = adapter_path
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], prefill: str | None) -> str:
        """Render messages to a single prompt string.

        Instruct models use Gemma's chat template. Base models have no chat
        template, so we fall back to a lightweight transcript format that the
        §3 prefill study relies on (it always supplies an explicit prefill).
        """

        if self.spec.is_base or self.tokenizer.chat_template is None:
            # Plain transcript; the prefill study controls exactly what the base
            # model continues from, so format fidelity matters less than
            # determinism. We mirror the "inline history" style validated in
            # Appendix A.3.
            parts = []
            for m in messages:
                tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
                parts.append(f"{tag}: {m.content}")
            parts.append("Assistant:")
            text = "\n".join(parts) + (" " + prefill if prefill else " ")
            return text

        # Gemma-3 has no system role; fold any system message into the first
        # user turn (standard practice for Gemma chat).
        chat = self._fold_system([{"role": m.role, "content": m.content} for m in messages])
        rendered = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            rendered += prefill
        return rendered

    @staticmethod
    def _fold_system(chat: list[dict]) -> list[dict]:
        if chat and chat[0]["role"] == "system":
            sys = chat[0]["content"]
            rest = chat[1:]
            if rest and rest[0]["role"] == "user":
                rest[0] = {"role": "user", "content": f"{sys}\n\n{rest[0]['content']}"}
                return rest
            return [{"role": "user", "content": sys}] + rest
        return chat

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: list[Message],
        *,
        n: int = 1,
        temperature: float = TEMPERATURE,
        max_new_tokens: int = MAX_NEW_TOKENS,
        prefill: str | None = None,
    ) -> list[str]:
        torch = self._torch
        prompt = self._render(messages, prefill)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=1.0,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        # Strip the prompt (and any prefill) -- callers want only the new text.
        completions = []
        for seq in out:
            new_tokens = seq[prompt_len:]
            completions.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )
        return completions

    # ------------------------------------------------------------------ #
    # Probing support (Appendix I)
    # ------------------------------------------------------------------ #
    def residual_stream(self, text: str):
        """Return per-layer hidden states for ``text`` (list over layers of
        ``[seq, d_model]`` tensors) plus the tokenizer offsets, for the
        logit-based emotion detector. See ``gnh.internal.emotion_logits``."""

        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return out.hidden_states, enc

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers
