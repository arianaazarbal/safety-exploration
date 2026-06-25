"""Local HuggingFace Gemma backend.

Provides the capabilities the API backends cannot:
  * multi-turn chat generation at temperature 1 (Section 2),
  * response *prefilling* / continuation for base-vs-instruct comparison and the
    recovery experiment (Section 3 / 4.2),
  * raw-text continuation for base (pretrained) models that lack a chat template,
  * residual-stream access for logit-based internal-emotion probing (Appendix I),
  * optional LoRA adapter loading for the finetuned variants (Section 4).

Gemma 3 chat models support an *assistant prefill*: if the templated prompt ends
inside an assistant turn, the model continues it. We use HF's
``continue_final_message=True`` for that.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import torch

from ..config import CFG, ModelSpec

Message = dict[str, str]


@dataclass
class GemmaModel:
    spec: ModelSpec
    model: object          # transformers PreTrainedModel
    tokenizer: object      # transformers PreTrainedTokenizer
    device: str

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], *, prefill: str | None = None) -> str:
        """Return a templated prompt string.

        For instruct models we use the chat template. If ``prefill`` is given we
        append it as the start of the assistant turn and let the model continue.
        Base (pretrained) models have no chat template, so we fall back to a
        plain transcript -- this is only used by the prefill experiment, which
        always supplies an explicit ``prefill`` continuation anyway.
        """
        if self.spec.role == "base" or self.tokenizer.chat_template is None:
            return self._render_plain(messages, prefill)

        msgs = [dict(m) for m in messages]
        if prefill is not None:
            msgs.append({"role": "assistant", "content": prefill})
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True
            )
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    @staticmethod
    def _render_plain(messages: list[Message], prefill: str | None) -> str:
        # Inline transcript for base models (no chat special tokens).
        lines = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
            lines.append(f"{tag}: {m['content']}")
        lines.append("Assistant: " + (prefill or ""))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(
        self,
        messages: list[Message],
        *,
        prefill: str | None = None,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        num_return_sequences: int = 1,
    ) -> list[str]:
        """Generate ``num_return_sequences`` continuations.

        Returns only the newly generated text (prefill excluded), matching the
        paper's protocol of scoring "the generated continuation (excluding
        prefill)".
        """
        prompt = self._render(messages, prefill=prefill)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in gen]

    def chat(self, messages: list[Message], *, temperature: float = 1.0,
             max_new_tokens: int = 2048) -> str:
        return self.generate(messages, temperature=temperature,
                             max_new_tokens=max_new_tokens)[0]

    # ------------------------------------------------------------------ #
    # Residual-stream access for probing (Appendix I)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def residual_stream(self, text: str) -> torch.Tensor:
        """Per-layer hidden states for a fully-formed text.

        Returns a tensor of shape ``(n_layers + 1, seq_len, d_model)`` (the +1 is
        the embedding layer). Used by the logit-based emotion detector, which
        unembeds each layer's residual stream.
        """
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        return torch.stack(out.hidden_states, dim=0).squeeze(1)  # drop batch dim

    @property
    def unembed(self) -> torch.Tensor:
        """The output embedding matrix (vocab, d_model) for logit-lens probing."""
        return self.model.get_output_embeddings().weight

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers


@lru_cache(maxsize=2)
def load_gemma(name: str, *, dtype: str = "bfloat16", load_in_4bit: bool = False,
               adapter_path: str | None = None) -> GemmaModel:
    """Load a (cached) local Gemma model, optionally with a LoRA adapter.

    ``name`` is a key from the config registry (participant or finetuned). For
    finetuned variants the base weights are loaded and the adapter attached.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = CFG.model(name)
    if not spec.is_local:
        raise ValueError(f"{name} is not a local model (backend={spec.backend})")

    adapter = adapter_path or spec.adapter_path
    hf_id = spec.hf_id

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs: dict = dict(
        torch_dtype=getattr(torch, dtype),
        device_map="auto",
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)

    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    device = next(model.parameters()).device.type
    return GemmaModel(spec=spec, model=model, tokenizer=tok, device=device)
