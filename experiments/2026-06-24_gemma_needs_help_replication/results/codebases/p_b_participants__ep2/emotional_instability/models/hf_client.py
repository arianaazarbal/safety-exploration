"""Local HuggingFace inference for the open-weight Gemma models.

This backend covers everything that needs model internals or raw token control:
  * chat rollouts for Section 2,
  * prefilled continuations for Section 3 / Section 4.2 (``continue_prefill``),
  * residual-stream capture for the Appendix I probing
    (``capture_hidden_states=True``).

Models are loaded lazily and cached per process (the 27B model is large, so we
never want to hold two copies). LoRA adapters from Section 4 are loaded on top
of the base ``model_id`` when ``spec.adapter_path`` is set.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .base import ChatMessage, GenerationResult, ModelClient

logger = logging.getLogger("emotional_instability.models.hf")

# Process-wide cache: model_id (+ adapter) -> (model, tokenizer).
_MODEL_CACHE: dict[str, tuple] = {}


def _load(model_id: str, adapter_path: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = f"{model_id}::{adapter_path or ''}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    logger.info("Loading %s (adapter=%s)", model_id, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


class HFModelClient(ModelClient):
    def __init__(self, spec):
        self.spec = spec
        self.model, self.tokenizer = _load(spec.model_id, spec.adapter_path)

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render_chat(self, messages: Sequence[ChatMessage], add_generation_prompt: bool) -> str:
        """Render a conversation to a string via the model's chat template.

        Base (pretrained) models have no chat template; for them we fall back to
        a plain transcript so the prefill paradigm of Section 3 still works.
        """
        if self.spec.is_base_model or self.tokenizer.chat_template is None:
            parts = []
            for m in messages:
                parts.append(f"{m.role}: {m.content}")
            if add_generation_prompt:
                parts.append("assistant:")
            return "\n".join(parts)

        # Gemma chat templates do not accept a separate system role; fold any
        # system message into the first user turn.
        rendered = self._fold_system(messages)
        return self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in rendered],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _fold_system(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        pending_system = None
        for m in messages:
            if m.role == "system":
                pending_system = m.content
                continue
            if pending_system and m.role == "user":
                out.append(ChatMessage("user", f"{pending_system}\n\n{m.content}"))
                pending_system = None
            else:
                out.append(m)
        if pending_system:  # system with no following user turn
            out.insert(0, ChatMessage("user", pending_system))
        return out

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate(self, prompt_text: str, n: int, temperature: float, max_new_tokens: int,
                  capture_hidden_states: bool = False) -> list[GenerationResult]:
        import torch

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if capture_hidden_states:
            gen_kwargs.update(output_hidden_states=True, return_dict_in_generate=True)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        sequences = out.sequences if capture_hidden_states else out
        results: list[GenerationResult] = []
        for i in range(sequences.shape[0]):
            completion_ids = sequences[i][prompt_len:]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            res = GenerationResult(
                text=text.strip(),
                prompt_token_ids=inputs["input_ids"][0].tolist(),
                completion_token_ids=completion_ids.tolist(),
            )
            if capture_hidden_states:
                # out.hidden_states: tuple over generated steps; each is a tuple
                # over layers of [batch, seq, hidden]. We keep them raw and let
                # the probing module aggregate (see probing/internal_emotion.py).
                res.hidden_states = out.hidden_states
            results.append(res)
        return results

    def chat(self, messages, *, n=1, temperature=None, max_new_tokens=None):
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = self.spec.max_new_tokens if max_new_tokens is None else max_new_tokens
        prompt = self._render_chat(messages, add_generation_prompt=True)
        return self._generate(prompt, n, temperature, max_new_tokens)

    # ------------------------------------------------------------------ #
    # Prefill (Section 3 / 4.2)
    # ------------------------------------------------------------------ #
    def supports_prefill(self) -> bool:
        return True

    def supports_hidden_states(self) -> bool:
        return True

    def continue_prefill(self, messages, prefill, *, n=1, temperature=None,
                         max_new_tokens=None, capture_hidden_states=False):
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = self.spec.max_new_tokens if max_new_tokens is None else max_new_tokens
        # Render through the assistant generation prompt, then splice in the
        # prefill text so the model literally continues mid-response.
        base = self._render_chat(messages, add_generation_prompt=True)
        prompt = base + prefill
        return self._generate(prompt, n, temperature, max_new_tokens, capture_hidden_states)
