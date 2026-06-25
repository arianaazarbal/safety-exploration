"""Local Gemma inference via HuggingFace transformers.

This backend powers everything that needs open weights:
  * Section 2 evaluation of Gemma-3-{12,27}B-it,
  * Section 3 base-vs-instruct prefilling (incl. Gemma-3-*-pt),
  * Section 4 finetuned variants (LoRA adapters loaded on top of the instruct
    weights),
  * Appendix I internal-emotion probing (exposes residual-stream hidden states).

Prefill is implemented by appending the partial assistant string to the rendered
chat template (with the closing turn marker omitted) and decoding only the newly
generated tokens.

Base (pretrained) models are not chat-tuned. As in the paper, we still drive them
through the chat template + prefill so that base and instruct continue from
identical starting points; the prefill is what lets a base model "continue the
response" (Section 3.1).
"""
from __future__ import annotations

from typing import Sequence

import config
from .base import ChatMessage, GenerationConfig, ModelClient


class HuggingFaceClient(ModelClient):
    supports_prefill = True

    def __init__(
        self,
        spec,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.spec_name = spec.name
        self.supports_prefill = spec.supports_prefill
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)

        model_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

        # Section 4 / Appendix I: a finetuned variant loads a LoRA adapter on top.
        if spec.adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, spec.adapter_path)

        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: Sequence[ChatMessage], prefill: str | None) -> str:
        """Render messages to a single prompt string.

        With a prefill we append the assistant generation prompt and then the
        partial assistant text, so decoding continues the same turn.
        """
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        is_base = self.spec.is_base

        if is_base:
            # Base checkpoints have no chat template; emit a plain transcript so
            # the prefill protocol still works. (See DESIGN.md, Section 3.)
            parts = [f"{m['role']}: {m['content']}" for m in msgs]
            parts.append("assistant: ")
            text = "\n".join(parts)
            if prefill:
                text += prefill
            return text

        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            text += prefill
        return text

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, messages: Sequence[ChatMessage], cfg: GenerationConfig) -> str:
        return self.generate_batch([messages], cfg)[0]

    def generate_batch(
        self, batch: Sequence[Sequence[ChatMessage]], cfg: GenerationConfig
    ) -> list[str]:
        torch = self._torch
        prompts = [self._render(m, cfg.prefill) for m in batch]

        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)

        do_sample = cfg.temperature and cfg.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=cfg.max_tokens,
                do_sample=do_sample,
                temperature=cfg.temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Strip the prompt tokens; decode only the continuation (prefill excluded).
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------ #
    # Hidden states (Appendix I logit-based emotion detection)
    # ------------------------------------------------------------------ #
    def forward_with_hidden_states(self, text: str):
        """Run a forward pass over `text`, returning per-layer residual streams.

        Returns (input_ids, hidden_states) where hidden_states is a tuple of
        (num_layers+1) tensors of shape [seq_len, d_model]. Used to unembed the
        residual stream and read off emotion-token logits.
        """
        torch = self._torch
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        hs = tuple(h.squeeze(0) for h in out.hidden_states)
        return enc["input_ids"].squeeze(0), hs

    def unembed(self, hidden: "Tensor"):  # noqa: F821
        """Project a residual-stream vector to vocab logits (final norm + lm_head)."""
        torch = self._torch
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        with torch.no_grad():
            normed = base.model.norm(hidden)
            return base.lm_head(normed)
