"""Local Gemma client via HuggingFace transformers.

Supports the three things the experiments need from an open-weights model that
the API path cannot give us:

  * chat-formatted multi-turn generation (Section 2),
  * assistant *prefill* / continuation (Section 3 prefill + Section 4 recovery),
  * hidden-state extraction for logit-based emotion probing (Appendix I).

A vLLM fast path is provided for plain batched chat generation (no prefill /
no hidden states); set ``backend_impl="vllm"``. The transformers path is the
default and the only one that supports prefill and hidden states.

LoRA adapters produced by the training scripts can be attached via
``adapter_path`` so finetuned checkpoints reuse the same client.
"""
from __future__ import annotations

import logging

from ..config import ModelSpec
from .base import GenerationConfig, Message

logger = logging.getLogger(__name__)


def render_base_prompt(messages: list[Message]) -> str:
    """Render a conversation as plain text for *base* (pretrained) models.

    Base models are not chat-tuned, so Section 3 feeds the history as text and
    relies on prefills. We use a simple, neutral transcript format and let the
    caller append the assistant prefill.
    """
    lines = []
    for m in messages:
        if m["role"] == "system":
            lines.append(m["content"])
        elif m["role"] == "user":
            lines.append(f"User: {m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"Assistant: {m['content']}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


class LocalGemmaClient:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        adapter_path: str | None = None,
        backend_impl: str = "transformers",
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.spec = spec
        self.adapter_path = adapter_path
        self.backend_impl = backend_impl
        self._device_map = device_map
        self._dtype = dtype
        self._model = None
        self._tokenizer = None
        self._vllm = None

    # ------------------------------------------------------------------ load
    def _ensure_loaded(self) -> None:
        if self.backend_impl == "vllm":
            self._ensure_vllm()
        else:
            self._ensure_transformers()

    def _ensure_transformers(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("loading %s (transformers)", self.spec.hf_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.hf_id,
            torch_dtype=getattr(torch, self._dtype),
            device_map=self._device_map,
            output_hidden_states=False,
        )
        if self.adapter_path:
            from peft import PeftModel

            logger.info("attaching LoRA adapter %s", self.adapter_path)
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    def _ensure_vllm(self) -> None:
        if self._vllm is not None:
            return
        from vllm import LLM

        if self.adapter_path:
            raise NotImplementedError(
                "Use the transformers backend to evaluate LoRA adapters, or merge "
                "the adapter first."
            )
        logger.info("loading %s (vLLM)", self.spec.hf_id)
        self._vllm = LLM(model=self.spec.hf_id, dtype=self._dtype)

    # ------------------------------------------------------------- prompting
    def _build_prompt(self, messages: list[Message], prefill: str | None) -> str:
        """Return the full prompt string the model should continue from."""
        if self.spec.kind == "base":
            prompt = render_base_prompt(messages)
            if prefill:
                prompt = prompt + " " + prefill
            return prompt

        # Instruct model: use the chat template, then splice the prefill on.
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            prompt = prompt + prefill
        return prompt

    # ------------------------------------------------------------- generate
    def supports_prefill(self) -> bool:
        return True

    def chat(self, messages: list[Message], cfg: GenerationConfig) -> str:
        self._ensure_loaded()
        if self.backend_impl == "vllm":
            return self._chat_vllm(messages, cfg)
        return self._chat_transformers(messages, cfg)

    def _chat_transformers(self, messages: list[Message], cfg: GenerationConfig) -> str:
        import torch

        if self.backend_impl == "vllm":  # safety
            raise RuntimeError("inconsistent backend")
        prompt = self._build_prompt(messages, cfg.prefill)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            top_p=cfg.top_p,
            pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        # Strip the prompt tokens so only the continuation remains (which, for a
        # prefilled call, excludes the prefill - matching Section 3 scoring).
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._apply_stop(text, cfg.stop)

    def _chat_vllm(self, messages: list[Message], cfg: GenerationConfig) -> str:
        from vllm import SamplingParams

        if cfg.prefill:
            raise NotImplementedError("Prefill requires the transformers backend.")
        prompt = self._build_prompt(messages, None)
        params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            stop=cfg.stop,
        )
        out = self._vllm.generate([prompt], params)
        return out[0].outputs[0].text

    @staticmethod
    def _apply_stop(text: str, stop: list[str] | None) -> str:
        if not stop:
            return text
        cut = len(text)
        for s in stop:
            i = text.find(s)
            if i != -1:
                cut = min(cut, i)
        return text[:cut]

    # ------------------------------------------------- hidden states (App. I)
    def hidden_states(self, messages: list[Message], prefill: str | None = None):
        """Return per-layer hidden states for the final position.

        Returns a tuple ``(hidden_states, tokenizer, model)`` where
        ``hidden_states`` is the tuple of layer activations from a single forward
        pass (no generation). Used by ``internal.logit_emotion``.
        """
        if not self.spec.supports_hidden_states:
            raise NotImplementedError(
                f"{self.spec.name} does not expose hidden states."
            )
        self._ensure_transformers()
        import torch

        prompt = self._build_prompt(messages, prefill)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        return out.hidden_states, self._tokenizer, self._model

    def close(self) -> None:
        self._model = None
        self._vllm = None
