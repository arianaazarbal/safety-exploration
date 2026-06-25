"""Local Gemma-3 inference via HuggingFace transformers (+ optional LoRA).

Handles both instruct (``-it``) and pretrained (``-pt``) checkpoints. Instruct
models use the tokenizer's chat template; base models receive a plain-text
transcript rendering (Section 3 prefilling relies on continuing prefilled text
rather than on chat formatting, which base models were never trained on).

Models are lazily loaded on first ``generate`` so that constructing a client is
cheap and importing this module does not require a GPU.
"""

from __future__ import annotations

from typing import Sequence

from .base import GenerationResult, Message

# Transcript markers used when rendering conversations for *base* models, which
# have no chat template. Kept minimal and neutral.
_BASE_ROLE_TAG = {"system": "System", "user": "User", "assistant": "Assistant"}


class GemmaLocalModel:
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        instruct: bool = True,
        dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
    ):
        self.name = name
        self.hf_id = hf_id
        self.instruct = instruct
        self.dtype = dtype
        self.default_max_new_tokens = max_new_tokens
        self.adapter_path = adapter_path
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self._model = None
        self._tokenizer = None

    # -- loading ----------------------------------------------------------
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.dtype)
        kwargs = dict(torch_dtype=dtype, device_map=self.device_map)
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        # Gemma-3 `-it` checkpoints are multimodal (Gemma3ForConditionalGeneration);
        # `-pt` text checkpoints load as causal LMs. Try the causal-LM head first,
        # then fall back to the image-text class (text-only inputs generate fine).
        try:
            model = AutoModelForCausalLM.from_pretrained(self.hf_id, **kwargs)
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(self.hf_id, **kwargs)

        if self.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model = model

    # -- prompt construction ---------------------------------------------
    def _build_inputs(self, messages: Sequence[Message], prefill: str | None):
        tok = self._tokenizer
        if self.instruct:
            # Gemma's chat template has no dedicated system role; fold a leading
            # system message into the first user turn.
            msgs = _fold_system(list(messages))
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = _render_base_transcript(messages)
        if prefill:
            prompt = prompt + prefill
        return tok(prompt, return_tensors="pt").to(self._model.device)

    # -- generation -------------------------------------------------------
    def generate(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
        prefill: str | None = None,
        stop: Sequence[str] | None = None,
    ) -> GenerationResult:
        import torch

        self._ensure_loaded()
        max_new_tokens = max_new_tokens or self.default_max_new_tokens
        inputs = self._build_inputs(messages, prefill)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self._tokenizer.pad_token_id
            or self._tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        # Decode only the newly generated tokens (continuation).
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        if stop:
            text = _apply_stop(text, stop)
        return GenerationResult(text=text, prefill=prefill or "")


def _fold_system(messages: list[Message]) -> list[Message]:
    if messages and messages[0]["role"] == "system":
        system = messages[0]["content"]
        rest = messages[1:]
        if rest and rest[0]["role"] == "user":
            merged = dict(rest[0])
            merged["content"] = f"{system}\n\n{rest[0]['content']}"
            return [merged] + rest[1:]
        return [{"role": "user", "content": system}] + rest
    return messages


def _render_base_transcript(messages: Sequence[Message]) -> str:
    """Plain-text transcript for base models (no chat template available)."""
    lines = []
    for m in messages:
        tag = _BASE_ROLE_TAG.get(m["role"], m["role"].title())
        lines.append(f"{tag}: {m['content']}")
    lines.append(f"{_BASE_ROLE_TAG['assistant']}:")
    return "\n".join(lines) + " "


def _apply_stop(text: str, stop: Sequence[str]) -> str:
    cut = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
