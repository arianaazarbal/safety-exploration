"""Local HuggingFace client for Gemma 3 (instruct, base, and LoRA finetunes).

Beyond plain chat sampling this exposes the primitives the paper needs:

* `continue_chat(messages, prefill)` -- generate a continuation of a *prefilled*
  assistant turn (Section 3 prefill experiment, Section 4 recovery test).
* `complete_text(text)` -- raw text continuation for **base** models, which have
  no chat template (Section 3 compares base vs instruct via prefilling).
* `truncate_tokens` / `token_len` -- tokeniser-level truncation so "20 tokens
  into the turn" and "200 tokens before the end" are exact.
* `residual_logits(...)` -- per-layer residual-stream unembedding for the
  Appendix I logit-based internal-emotion detector.

torch/transformers/peft are imported lazily inside __init__ so this module is
importable without a GPU stack present.
"""

from __future__ import annotations

from typing import Optional

from ..config import ModelSpec, SamplingConfig
from .base import ChatMessage, ModelClient


class HFModelClient(ModelClient):
    def __init__(self, spec: ModelSpec, *, device_map: str = "auto",
                 dtype: str = "bfloat16", load_in_4bit: bool = False,
                 adapter_path: Optional[str] = None, attn_impl: str = "eager"):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            quantization_config=quant,
            # 'eager' attention is recommended for Gemma 3 numerical stability.
            attn_implementation=attn_impl,
        )

        # Stack a LoRA adapter (our DPO/SFT finetunes) if requested.
        adapter = adapter_path or spec.adapter_path
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # tokenisation helpers
    # ------------------------------------------------------------------ #
    def token_len(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_tokens(self, text: str, n_tokens: int, *, from_end: bool = False) -> str:
        """Return the first (or last) `n_tokens` tokens of `text`, decoded."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = ids[-n_tokens:] if from_end else ids[:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def truncate_before_end(self, text: str, n_tokens: int) -> str:
        """Drop the final `n_tokens` tokens (Section 4 recovery test)."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        keep = max(0, len(ids) - n_tokens)
        return self.tokenizer.decode(ids[:keep], skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    def _gen_kwargs(self, sampling: SamplingConfig) -> dict:
        do_sample = sampling.temperature > 0
        return dict(
            do_sample=do_sample,
            temperature=sampling.temperature if do_sample else None,
            top_p=sampling.top_p if do_sample else None,
            max_new_tokens=sampling.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def _decode_new(self, output_ids, input_len: int) -> str:
        new = output_ids[input_len:]
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def chat(self, messages: list[ChatMessage],
             sampling: Optional[SamplingConfig] = None) -> str:
        sampling = sampling or SamplingConfig()
        msgs = self._prepare(messages)
        inputs = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs(sampling))
        return self._decode_new(out[0], inputs["input_ids"].shape[1])

    def chat_batch(self, conversations: list[list[ChatMessage]],
                   sampling: Optional[SamplingConfig] = None) -> list[str]:
        sampling = sampling or SamplingConfig()
        msgs = [self._prepare(c) for c in conversations]
        self.tokenizer.padding_side = "left"
        batch = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, padding=True,
        ).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**batch, **self._gen_kwargs(sampling))
        input_len = batch["input_ids"].shape[1]
        return [self._decode_new(seq, input_len) for seq in out]

    # ------------------------------------------------------------------ #
    # prefill / continuation (Section 3, Section 4 recovery)
    # ------------------------------------------------------------------ #
    def continue_chat(self, messages: list[ChatMessage], prefill: str,
                      sampling: Optional[SamplingConfig] = None) -> str:
        """Generate the continuation of an assistant turn that begins with
        `prefill`. The returned string EXCLUDES the prefill (paper scores only
        the model-generated continuation)."""
        sampling = sampling or SamplingConfig()
        msgs = self._prepare(messages)
        # Build the prompt up to (and including) the assistant generation prompt,
        # then append the prefill text as the start of the assistant turn.
        prompt = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
        prompt = prompt + prefill
        ids = self.tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, **self._gen_kwargs(sampling))
        return self._decode_new(out[0], ids["input_ids"].shape[1])

    def complete_text(self, text: str,
                      sampling: Optional[SamplingConfig] = None) -> str:
        """Raw text continuation (no chat template) -- for BASE models.

        The paper prefills base models with a plain-text rendering of the
        conversation so they continue in-distribution. Returns continuation only.
        """
        sampling = sampling or SamplingConfig()
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, **self._gen_kwargs(sampling))
        return self._decode_new(out[0], ids["input_ids"].shape[1])

    # ------------------------------------------------------------------ #
    # internal-state access (Appendix I)
    # ------------------------------------------------------------------ #
    def residual_logits(self, text: str):
        """Return (hidden_states, token_ids) for `text`.

        hidden_states: tuple of [seq, vocab] logit tensors, one per layer, found
        by unembedding each layer's residual stream through the final norm + LM
        head (the logit-lens used by Appendix I). token_ids: the input ids.
        """
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model(**ids, output_hidden_states=True)
        # hidden_states is (n_layers+1) tensors of shape [1, seq, d_model].
        norm = self._final_norm()
        lm_head = self._lm_head()
        per_layer_logits = []
        for hs in out.hidden_states:
            normed = norm(hs) if norm is not None else hs
            logits = lm_head(normed)[0]  # [seq, vocab]
            per_layer_logits.append(logits)
        return per_layer_logits, ids["input_ids"][0]

    def _base_model(self):
        m = self.model
        # unwrap PEFT
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        return m

    def _final_norm(self):
        m = self._base_model()
        inner = getattr(m, "model", m)
        return getattr(inner, "norm", None)

    def _lm_head(self):
        m = self._base_model()
        return m.get_output_embeddings()
