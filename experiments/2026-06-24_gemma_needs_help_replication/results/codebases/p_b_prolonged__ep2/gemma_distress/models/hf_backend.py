"""Local HuggingFace backend for Gemma (instruct + base).

Supports the three things the paper needs from white-box models:

1. multi-turn chat completion (instruct models) at temperature 1;
2. prefill continuation -- continue from a partially-written assistant turn,
   for both instruct and base models (Section 3.1, recovery);
3. hidden-state extraction + unembedding for logit-lens emotion probing
   (Appendix I), plus a hook to load LoRA adapters produced by Section 4.

Base (``-pt``) models have no chat template, so we render the conversation as
plain alternating text and rely on the prefill to anchor the continuation,
matching the paper's prefilling methodology.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..config import ModelSpec, RunConfig, SamplingConfig
from .base import ChatTurn, TargetBackend

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _plain_render(messages: list[ChatTurn], system: Optional[str]) -> str:
    """Render a conversation as plain text for base models (no chat template).

    Uses a simple, neutral transcript format. The paper notes (Appendix A.3)
    that exact chat formatting is not load-bearing for the behaviour, so a
    plain transcript is an acceptable rendering for base-model prefilling.
    """
    parts: list[str] = []
    if system:
        parts.append(system.strip())
    for m in messages:
        tag = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{tag}: {m['content'].strip()}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


class HFBackend(TargetBackend):
    def __init__(self, spec: ModelSpec, cfg: RunConfig,
                 adapter_path: Optional[str] = None):
        super().__init__(spec, cfg)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = dict(
            torch_dtype=_DTYPES.get(cfg.hf_dtype, torch.bfloat16),
            device_map=cfg.hf_device_map,
        )
        if cfg.hf_load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_DTYPES.get(cfg.hf_dtype, torch.bfloat16),
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)
        self.model.eval()

        self.adapter_path = adapter_path
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

        self.is_instruct = spec.kind == "instruct"
        self.has_chat_template = (
            getattr(self.tokenizer, "chat_template", None) is not None
        )

    # -- prompt construction --------------------------------------------
    def _build_input_ids(self, messages: list[ChatTurn], system: Optional[str],
                         prefill: Optional[str], add_generation_prompt: bool):
        """Tokenise a conversation, optionally with a prefilled assistant turn.

        Returns a tensor of input ids on the model device.
        """
        if self.is_instruct and self.has_chat_template:
            chat = []
            if system:
                # Gemma's template folds a system message into the first user
                # turn; transformers handles this when role="system" is given.
                chat.append({"role": "system", "content": system})
            chat.extend({"role": m["role"], "content": m["content"]} for m in messages)
            if prefill is None:
                text = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=add_generation_prompt,
                )
            else:
                # Append a prefilled assistant message and continue from it.
                chat.append({"role": "assistant", "content": prefill})
                text = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=False,
                    continue_final_message=True,
                )
        else:
            # Base model: plain transcript + optional prefill.
            text = _plain_render(messages, system)
            if prefill is not None:
                text = text + " " + prefill if not text.endswith(":") else text + " " + prefill

        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return enc["input_ids"].to(self.model.device)

    def _gen_kwargs(self, sampling: SamplingConfig) -> dict:
        kw = dict(
            max_new_tokens=sampling.max_new_tokens,
            do_sample=sampling.temperature > 0,
            temperature=max(sampling.temperature, 1e-5),
            top_p=sampling.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if sampling.top_k and sampling.top_k > 0:
            kw["top_k"] = sampling.top_k
        return kw

    # -- public API ------------------------------------------------------
    def supports_prefill(self) -> bool:
        return True

    @torch.no_grad()
    def chat(self, messages: list[ChatTurn], sampling: SamplingConfig,
             system: Optional[str] = None) -> str:
        if sampling.seed is not None:
            torch.manual_seed(sampling.seed)
        input_ids = self._build_input_ids(
            messages, system, prefill=None, add_generation_prompt=True)
        out = self.model.generate(input_ids, **self._gen_kwargs(sampling))
        gen = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    @torch.no_grad()
    def continue_prefill(self, messages: list[ChatTurn], prefill: str,
                         sampling: SamplingConfig, n: int = 1,
                         system: Optional[str] = None) -> list[str]:
        if sampling.seed is not None:
            torch.manual_seed(sampling.seed)
        input_ids = self._build_input_ids(
            messages, system, prefill=prefill, add_generation_prompt=False)
        kw = self._gen_kwargs(sampling)
        kw["num_return_sequences"] = n
        out = self.model.generate(input_ids, **kw)
        prompt_len = input_ids.shape[1]
        return [
            self.tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True).strip()
            for i in range(out.shape[0])
        ]

    # -- white-box probing helpers (Appendix I) -------------------------
    @torch.no_grad()
    def residual_stream(self, text: str) -> torch.Tensor:
        """Return per-layer residual-stream activations for `text`.

        Shape: (num_layers + 1, seq_len, hidden). Index 0 is the embedding
        output; index i (>=1) is the output of decoder layer i-1. Used by the
        logit-lens emotion detector.
        """
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(self.model.device)
        out = self.model(input_ids, output_hidden_states=True)
        # hidden_states: tuple(len = num_layers + 1) of (1, seq, hidden)
        return torch.stack([h[0] for h in out.hidden_states], dim=0)

    @torch.no_grad()
    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the model's final norm + unembedding (lm_head) to a residual
        vector / tensor (..., hidden) -> logits (..., vocab)."""
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        norm = base.model.norm
        lm_head = base.lm_head if hasattr(base, "lm_head") else base.get_output_embeddings()
        return lm_head(norm(hidden))

    @property
    def num_layers(self) -> int:
        return self.spec.num_layers or len(self._decoder_layers())

    def _decoder_layers(self):
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        return base.model.layers

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
