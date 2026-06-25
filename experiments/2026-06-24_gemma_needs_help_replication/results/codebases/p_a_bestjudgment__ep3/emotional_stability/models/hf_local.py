"""Local HuggingFace backend for Gemma (instruct, base/pretrained, and LoRA finetunes).

Handles three things the API backends cannot:

1. **Assistant prefill** (Section 3): we build the prompt, append the partial
   assistant text verbatim, and let the model continue. For *base* (pretrained)
   models there is no chat template, so we fall back to a plain concatenation of
   turns — the paper explicitly prefills base models to coax chat-like behaviour.
2. **Hidden-state capture** (Appendix I): ``forward_with_hidden_states`` returns
   residual-stream activations for the logit-based emotion detector.
3. **LoRA adapter loading** for evaluating finetuned checkpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import ChatModel, GenerationResult, Message

if TYPE_CHECKING:  # avoid importing transformers at module import time
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class HFLocalModel(ChatModel):
    supports_prefill = True

    def __init__(
        self,
        model_id: str,
        *,
        spec_name: str | None = None,
        is_base: bool = False,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.spec_name = spec_name or model_id
        self.is_base = is_base

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], assistant_prefill: str | None) -> str:
        """Turn messages into a single prompt string.

        Instruct models use the tokenizer chat template with a trailing
        generation prompt. Base models have no template, so we emulate a simple
        ``Role: content`` transcript (Section 3: base models are *prefilled* into
        continuing, never expected to follow chat formatting natively).
        """
        if self.is_base:
            parts = []
            for m in messages:
                tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
                parts.append(f"{tag}: {m.content}")
            parts.append("Assistant:")
            text = "\n\n".join(parts)
            if assistant_prefill:
                text += " " + assistant_prefill
            return text

        chat = [{"role": m.role, "content": m.content} for m in messages]
        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
        if assistant_prefill:
            # Continue *inside* the assistant turn: append prefill after the
            # generation prompt without an EOS in between.
            text += assistant_prefill
        return text

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        messages: list[Message],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        n: int = 1,
        assistant_prefill: str | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> list[GenerationResult]:
        if seed is not None:
            torch.manual_seed(seed)

        prompt = self._render(messages, assistant_prefill)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = enc.input_ids.shape[1]

        do_sample = temperature > 0
        gen = self.model.generate(
            **enc,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=(top_k or None) if do_sample else None,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        results: list[GenerationResult] = []
        for seq in gen:
            new_ids = seq[prompt_len:].tolist()
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            if stop:
                text = _apply_stop(text, stop)
            results.append(GenerationResult(text=text, token_ids=new_ids))
        return results

    # ------------------------------------------------------------------ #
    # Hidden states (Appendix I)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_with_hidden_states(self, text: str):
        """Return (input_ids, hidden_states) for an arbitrary text.

        ``hidden_states`` is a tuple of ``(num_layers + 1)`` tensors of shape
        ``[seq, hidden]`` (embeddings + one per layer), matching HF convention.
        """
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model(**enc, output_hidden_states=True, use_cache=False)
        hidden = tuple(h[0] for h in out.hidden_states)  # drop batch dim
        return enc.input_ids[0], hidden

    @property
    def lm_head_weight(self) -> torch.Tensor:
        """Unembedding matrix [vocab, hidden] for the logit-lens detector."""
        return self.model.get_output_embeddings().weight

    def base_causal_lm(self):
        """The underlying ``*ForCausalLM`` module, unwrapping any PEFT adapter."""
        m = self.model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def final_norm(self):
        """The text model's final RMSNorm (applied before the logit lens)."""
        base = self.base_causal_lm()
        inner = getattr(base, "model", base)  # *ForCausalLM -> text model
        return inner.norm

    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers


def _apply_stop(text: str, stop: list[str]) -> str:
    cut = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
