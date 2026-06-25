"""Local HuggingFace inference backend for Gemma (and any HF chat model).

Supports the three capabilities the paper needs:
  * multi-turn ``chat``
  * assistant ``continue_from`` (prefill) — used by Sections 3 & 4
  * ``residual_logit_lens`` for Appendix I internal-emotion detection

Base (pretrained) Gemma has no chat template; for it we render the conversation
with a lightweight role-tagged template so that prefilling works consistently
(see ``_render_base``). This mirrors the paper's approach of using prefills so
base models "consistently continue the model response" (Section 3.1).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..config import ModelConfig
from .base import GenerationResult, Message

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HFChatModel:
    def __init__(self, cfg: ModelConfig, device_map: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.name = cfg.name
        self.family = cfg.family
        self.variant = cfg.variant
        self.is_base = cfg.variant == "base"

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_id,
            torch_dtype=_DTYPES.get(cfg.dtype, torch.bfloat16),
            device_map=device_map,
        )
        if cfg.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, cfg.adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], *, add_generation_prompt: bool,
                prefill: str = "") -> str:
        """Return the full prompt string the model should continue from."""
        if self.is_base:
            text = self._render_base(messages, add_generation_prompt)
        else:
            text = self.tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        return text + prefill

    @staticmethod
    def _render_base(messages: list[Message], add_generation_prompt: bool) -> str:
        # Minimal role-tagged rendering for pretrained models that lack a chat
        # template. Kept deliberately plain so behaviour is driven by prefills.
        parts = []
        for m in messages:
            parts.append(f"{m.role.capitalize()}: {m.content}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n\n".join(parts) + (" " if add_generation_prompt else "")

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _generate(self, prompt_text: str, temperature: float,
                  max_new_tokens: int) -> GenerationResult:
        enc = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        do_sample = temperature and temperature > 0
        out = self.model.generate(
            **enc,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][enc["input_ids"].shape[1]:]
        tokens = self.tokenizer.convert_ids_to_tokens(gen_ids)
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(text=text, tokens=tokens)

    def chat(self, messages, *, temperature=1.0, max_new_tokens=None) -> GenerationResult:
        prompt = self._render(messages, add_generation_prompt=True)
        return self._generate(prompt, temperature, max_new_tokens or self.cfg.max_new_tokens)

    def continue_from(self, messages, prefill, *, temperature=1.0,
                      max_new_tokens=None) -> GenerationResult:
        prompt = self._render(messages, add_generation_prompt=True, prefill=prefill)
        return self._generate(prompt, temperature, max_new_tokens or self.cfg.max_new_tokens)

    # ------------------------------------------------------------------ #
    # Appendix I: residual-stream logit lens
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def residual_logit_lens(self, text: str) -> torch.Tensor:
        """Return a (num_layers, seq_len, vocab) tensor of unembedded residual
        streams: for each layer's hidden state, apply the final norm + LM head.

        Used by ``internal/logit_emotion.py`` to read emotion-token logits at
        each depth, per Appendix I.
        """
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        out = self.model(**enc, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple: (num_layers+1) x (1, seq, d)

        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        norm = base.model.norm
        lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()

        logits_per_layer = []
        for hs in hidden_states[1:]:           # skip embedding layer
            normed = norm(hs)
            logits = lm_head(normed)           # (1, seq, vocab)
            logits_per_layer.append(logits[0])
        return torch.stack(logits_per_layer, dim=0)  # (layers, seq, vocab)

    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers
