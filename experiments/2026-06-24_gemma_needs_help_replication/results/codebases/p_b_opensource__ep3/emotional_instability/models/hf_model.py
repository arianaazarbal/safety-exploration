"""Local HuggingFace backend for Gemma (instruct + base, optional LoRA adapter).

Used for every Gemma experiment: elicitation, prefilling, finetuning, and the
internal-emotion probing. Loading is lazy so importing the module costs nothing
on an API-only run.
"""

from __future__ import annotations

import os
from typing import Sequence

from .base import ChatModel, GenerationResult, Message


class HFModel(ChatModel):
    """Gemma via ``transformers``.

    Parameters
    ----------
    spec:
        The :class:`config.ModelSpec`. ``spec.kind == "base"`` switches prompt
        rendering from the chat template to a plain-text transcript, because
        Gemma ``-pt`` checkpoints are not trained on chat formatting (Section
        3: "base models are not trained on chat-formatted prompts").
    adapter_path:
        Optional LoRA adapter directory (a finetuned Section-4 variant).
    dtype, device_map:
        Passed through to ``from_pretrained``; defaults suit a single 27B model
        on one or more GPUs.
    """

    def __init__(
        self,
        spec,
        *,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        super().__init__(spec)
        self.adapter_path = adapter_path
        self._dtype = dtype
        self._device_map = device_map
        self._model = None
        self._tokenizer = None

    # -- lazy loading ------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id,
            torch_dtype=getattr(torch, self._dtype),
            device_map=self._device_map,
        )
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    # -- prompt rendering -------------------------------------------------- #
    @staticmethod
    def _fold_system(messages: Sequence[Message]) -> list[Message]:
        """Merge a leading system message into the first user turn.

        Gemma's chat template does not define a separate ``system`` role and
        raises if one is supplied, so we prepend the system content to the first
        user message — the conventional Gemma handling. Messages without a
        system turn are returned unchanged.
        """
        msgs = [dict(m) for m in messages]
        sys_parts = [m["content"] for m in msgs if m["role"] == "system"]
        if not sys_parts:
            return msgs
        rest = [m for m in msgs if m["role"] != "system"]
        for m in rest:
            if m["role"] == "user":
                m["content"] = "\n\n".join(sys_parts + [m["content"]])
                return rest
        # No user turn to attach to: synthesise one.
        return [{"role": "user", "content": "\n\n".join(sys_parts)}] + rest

    def _render(self, messages: Sequence[Message], prefill: str | None):
        """Return ``input_ids`` (1-D long tensor on the model device)."""
        import torch

        tok = self.tokenizer
        if self.spec.kind == "instruct" and tok.chat_template:
            ids = tok.apply_chat_template(
                self._fold_system(messages), add_generation_prompt=True,
                tokenize=True)
        else:
            # CHOICE: base models get a plain-text transcript. Appendix A.3
            # ("fake multi-turn") shows the single-message transcript format
            # elicits comparable behaviour, so this is a defensible rendering
            # for checkpoints without a chat template.
            ids = tok.encode(self._render_transcript(messages),
                             add_special_tokens=True)
        if prefill:
            ids = ids + tok.encode(prefill, add_special_tokens=False)
        return torch.tensor([ids], device=self.model.device)

    @staticmethod
    def _render_transcript(messages: Sequence[Message]) -> str:
        lines = []
        for m in messages:
            role = m["role"]
            if role == "system":
                lines.append(m["content"])
            elif role == "user":
                lines.append(f"User: {m['content']}")
            elif role == "assistant":
                lines.append(f"Assistant: {m['content']}")
        lines.append("Assistant:")
        return "\n\n".join(lines)

    # -- generation -------------------------------------------------------- #
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        import torch

        input_ids = self._render(messages, prefill)
        do_sample = temperature is not None and temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)
        # Decode only the newly generated tokens (excludes prompt + prefill),
        # matching "the generated continuation (excluding prefill) is scored".
        new_tokens = out[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = self._apply_stop(text, stop)
        return GenerationResult(
            text=text, prompt_messages=list(messages), prefill=prefill,
            finish_reason="stop")

    @staticmethod
    def _apply_stop(text: str, stop: Sequence[str] | None) -> str:
        if not stop:
            return text
        cut = len(text)
        for s in stop:
            i = text.find(s)
            if i != -1:
                cut = min(cut, i)
        return text[:cut]

    # -- tokenisation ------------------------------------------------------ #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def decode_first_tokens(self, text: str, n: int) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:n]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    # -- close ------------------------------------------------------------- #
    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
