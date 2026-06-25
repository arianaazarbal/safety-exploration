"""Local HuggingFace chat model for Gemma (instruct + base/pretrained).

Supports:
  * standard multi-turn chat generation via the tokenizer chat template;
  * assistant *prefill* (force-continue a partial assistant turn) for the
    Section 3 base-vs-instruct experiment -- this is why the prefill experiment
    is local-only;
  * loading a LoRA adapter on top of the base weights (to evaluate DPO/SFT
    finetunes from Section 4);
  * base (`-pt`) models, which have no chat template: we format the
    conversation into plain text ourselves.

Designed to be importable without torch/transformers installed; those are only
required when an HFChatModel is actually constructed.
"""

from __future__ import annotations

from ..config import DEFAULT_MAX_TOKENS, ModelSpec
from .base import GenerationResult, Message


class HFChatModel:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        adapter_path: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ):
        self.spec = spec
        self.key = spec.key
        self.supports_prefill = spec.supports_prefill
        self.adapter_path = adapter_path
        self._device_map = device_map
        self._dtype = dtype
        self._attn = attn_implementation
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    # Lazy loading
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self._dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        load_kwargs = dict(torch_dtype=dtype, device_map=self._device_map)
        if self._attn:
            load_kwargs["attn_implementation"] = self._attn
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id, **load_kwargs
        )
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _render(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool,
        continue_final_message: bool = False,
    ) -> str:
        """Render chat messages to a prompt string.

        Instruct models use the tokenizer chat template. Base (`-pt`) models
        have no template, so we fall back to a simple, explicit text format that
        still presents the turn structure (this matches the paper's approach of
        prefilling base models so they "consistently continue the response").
        """
        tok = self._tokenizer
        if self.spec.is_base or tok.chat_template is None:
            return _plain_text_format(
                messages,
                add_generation_prompt=add_generation_prompt,
                continue_final=continue_final_message,
            )
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate_from_text(
        self, prompt_text: str, *, temperature: float, max_tokens: int,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        import torch

        self._ensure_loaded()
        tok = self._tokenizer
        inputs = tok(prompt_text, return_tensors="pt").to(self._model.device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen_tokens, skip_special_tokens=True)
        if stop:
            text = _truncate_at_stop(text, stop)
        return GenerationResult(text=text, finish_reason="stop")

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        prompt = self._render(messages, add_generation_prompt=True)
        return self._generate_from_text(
            prompt, temperature=temperature, max_tokens=max_tokens, stop=stop
        )

    def prefill(
        self,
        messages: list[Message],
        prefill_text: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> GenerationResult:
        """Force the assistant to continue ``prefill_text`` and return ONLY the
        generated continuation (excluding the prefill)."""
        convo = list(messages) + [{"role": "assistant", "content": prefill_text}]
        prompt = self._render(
            convo, add_generation_prompt=False, continue_final_message=True
        )
        return self._generate_from_text(
            prompt, temperature=temperature, max_tokens=max_tokens
        )

    # ------------------------------------------------------------------ #
    # Token utilities (used by the prefill experiment for truncation)
    # ------------------------------------------------------------------ #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def truncate_from_end_tokens(self, text: str, n_from_end: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if n_from_end >= len(ids):
            return ""
        return self.tokenizer.decode(ids[:-n_from_end], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _plain_text_format(
    messages: list[Message], *, add_generation_prompt: bool, continue_final: bool
) -> str:
    """Plain-text rendering for base models (no chat template).

    Uses simple Role: prefixes. When continuing a final assistant message we
    emit its content without a trailing newline so generation continues it.
    """
    parts: list[str] = []
    role_label = {"system": "System", "user": "User", "assistant": "Assistant"}
    for i, m in enumerate(messages):
        label = role_label.get(m["role"], m["role"].capitalize())
        is_last = i == len(messages) - 1
        if is_last and continue_final and m["role"] == "assistant":
            parts.append(f"{label}: {m['content']}")  # no newline: continue it
            return "\n\n".join(parts)
        parts.append(f"{label}: {m['content']}")
    text = "\n\n".join(parts)
    if add_generation_prompt:
        text += "\n\nAssistant:"
    return text


def _truncate_at_stop(text: str, stop: list[str]) -> str:
    cut = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
