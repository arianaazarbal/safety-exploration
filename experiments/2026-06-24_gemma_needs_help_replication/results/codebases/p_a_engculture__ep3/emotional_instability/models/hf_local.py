"""HuggingFace transformers backend.

Used for the two experiments that need lower-level access than vLLM exposes:

  * Section 3 prefill: continue a *forced* assistant prefix. For instruct models
    we render the chat template with ``continue_final_message=True``; for base
    (pretrained) models, which have no chat template, we build a plain-text
    transcript and let the model continue it (the paper's approach for making
    base models "consistently continue the model response").

  * Appendix I probing: ``residual_streams`` returns per-layer hidden states so
    the logit-lens emotion detector can unembed them.
"""
from __future__ import annotations

import torch

from .base import ChatMessage, GenerationResult, SamplingParams


# Plain-text transcript format for base models (no chat template available).
_BASE_ROLE_TAG = {"system": "System", "user": "User", "assistant": "Assistant"}


class HFLocalClient:
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        is_base: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.is_base = is_base
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype), device_map=device_map
        )
        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ---- prompt construction -------------------------------------------------
    def _render_prompt(self, messages: list[ChatMessage], prefill: str | None) -> str:
        if self.is_base:
            lines = [f"{_BASE_ROLE_TAG[m.role]}: {m.content}" for m in messages]
            lines.append(f"{_BASE_ROLE_TAG['assistant']}: {prefill or ''}")
            # Strip the trailing space when no prefill, so generation starts clean.
            return "\n".join(lines).rstrip() if prefill is None else "\n".join(lines)

        msgs = [m.as_dict() for m in messages]
        if prefill is not None:
            # Force the model to continue an assistant turn we supply.
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True
            )
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # ---- generation ----------------------------------------------------------
    @torch.no_grad()
    def _generate_text(self, prompt: str, params: SamplingParams) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        if params.seed is not None:
            torch.manual_seed(params.seed)
        out = self.model.generate(
            **inputs,
            do_sample=params.temperature > 0,
            temperature=max(params.temperature, 1e-6),
            top_p=params.top_p,
            max_new_tokens=params.max_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            completion_tokens=int(gen_ids.shape[0]),
        )

    def generate(self, messages: list[ChatMessage], params: SamplingParams) -> GenerationResult:
        return self._generate_text(self._render_prompt(messages, None), params)

    def generate_with_prefill(
        self, messages: list[ChatMessage], prefill: str, params: SamplingParams
    ) -> GenerationResult:
        """Continue ``prefill`` as the assistant turn; returns only the continuation."""
        return self._generate_text(self._render_prompt(messages, prefill), params)

    def generate_batch(
        self, conversations: list[list[ChatMessage]], params: SamplingParams
    ) -> list[GenerationResult]:
        return [self.generate(c, params) for c in conversations]

    # ---- probing support (Appendix I) ---------------------------------------
    @torch.no_grad()
    def residual_streams(self, text: str) -> tuple[torch.Tensor, list[int]]:
        """Return per-layer residual-stream activations for ``text``.

        Output shape: ``(num_layers + 1, seq_len, d_model)`` (embedding output
        plus one per decoder layer), and the token ids. The logit-lens detector
        unembeds these against the emotion-token vocabulary.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model(**inputs, output_hidden_states=True)
        hidden = torch.stack(out.hidden_states, dim=0).squeeze(1)  # (L+1, T, d)
        return hidden, inputs["input_ids"][0].tolist()

    @torch.no_grad()
    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply final norm + LM head to residual-stream vectors -> vocab logits."""
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        norm = base.model.norm
        lm_head = base.get_output_embeddings()
        return lm_head(norm(hidden))
