"""Local Hugging Face transformers backend for Gemma.

Serves four needs:
  * chat generation for elicitation / Petri / capabilities (`chat`, `chat_batch`),
  * raw continuation for the §3 prefill experiment (`continue_text`),
  * LoRA-adapter loading for the DPO/SFT finetuned variants,
  * residual-stream access for the Appendix-I probing (`forward_hidden_states`).

Gemma instruct models carry a chat template; base ("-pt") models do not, so we
expose the raw-continuation path and skip templating for them.

A vLLM backend (models/vllm_model.py) mirrors `chat`/`continue_text` for fast
sampling, but probing and training require this transformers backend.
"""
from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec
from ..utils.io import get_env
from ..utils.logging import get_logger
from .base import ChatModel, Generation, Message, SamplingParams

log = get_logger("models.hf")

_DTYPES = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}


class HFModel(ChatModel):
    supports_chat = True

    def __init__(self, spec: ModelSpec, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = spec.name
        self.family = spec.family
        self.kind = spec.kind
        self.spec = spec
        self.supports_continuation = True  # local models can always raw-continue
        token = get_env("HF_TOKEN", required=False)

        log.info("Loading %s (%s)", spec.hf_id, spec.kind)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            torch_dtype=getattr(torch, _DTYPES[spec.dtype]),
            device_map=device_map,
            token=token,
        )

        if spec.adapter_path:
            from peft import PeftModel

            log.info("Attaching LoRA adapter %s", spec.adapter_path)
            self.model = PeftModel.from_pretrained(self.model, spec.adapter_path)
        self.model.eval()

        # Base ("-pt") models have no chat template; flag for the protocol layer.
        self.has_chat_template = (
            spec.kind != "base" and self.tokenizer.chat_template is not None
        )

    # --- prompt construction ----------------------------------------------
    def _render_chat(self, messages: Sequence[Message]) -> str:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def _generate(self, prompts: list[str], params: SamplingParams) -> list[str]:
        import torch

        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(self.model.device)
        do_sample = params.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=do_sample,
                temperature=params.temperature if do_sample else None,
                top_p=params.top_p if do_sample else None,
                max_new_tokens=params.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    # --- chat --------------------------------------------------------------
    def chat(self, messages: Sequence[Message], params: SamplingParams) -> Generation:
        return self.chat_batch([messages], params)[0]

    def chat_batch(
        self, batch: Sequence[Sequence[Message]], params: SamplingParams
    ) -> list[Generation]:
        if not self.has_chat_template:
            raise RuntimeError(
                f"{self.name} is a base model without a chat template; use "
                f"continue_text() / the prefill experiment instead."
            )
        prompts = [self._render_chat(m) for m in batch]
        texts = self._generate(prompts, params)
        return [
            Generation(text=t, prompt_messages=tuple(m), finish_reason="stop")
            for t, m in zip(texts, batch)
        ]

    # --- raw continuation (prefill experiment) -----------------------------
    def continue_text(self, prefix: str, params: SamplingParams) -> Generation:
        return self.continue_text_batch([prefix], params)[0]

    def continue_text_batch(
        self, prefixes: Sequence[str], params: SamplingParams
    ) -> list[Generation]:
        texts = self._generate(list(prefixes), params)
        return [Generation(text=t, finish_reason="stop") for t in texts]

    # --- token utilities (used by prefill truncation) ----------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_to_tokens(self, text: str, n_tokens: int, from_end: bool = False) -> str:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = ids[-n_tokens:] if from_end else ids[:n_tokens]
        return self.tokenizer.decode(ids)

    # --- probing (Appendix I) ---------------------------------------------
    def forward_hidden_states(self, text: str):
        """Return (token_ids, hidden_states) for residual-stream probing.

        hidden_states is a tuple of length (n_layers+1), each [1, seq, d_model],
        as returned by transformers with output_hidden_states=True.
        """
        import torch

        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        return enc["input_ids"][0], out.hidden_states

    def lm_head_weight(self):
        """Unembedding matrix (for the logit-lens probe)."""
        return self.model.get_output_embeddings().weight
