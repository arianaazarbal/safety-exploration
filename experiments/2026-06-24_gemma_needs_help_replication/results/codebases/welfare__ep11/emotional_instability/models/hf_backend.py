"""HuggingFace `transformers` backend for local Gemma models.

Used for:
  * base (pretrained) Gemma -- which has no chat template, so we drive it with
    explicit prefilling (Section 3);
  * our LoRA finetunes (Section 4) -- the adapter is loaded on top of the
    instruct base;
  * hidden-state extraction (Appendix I internal-emotion probing).

For high-throughput plain generation of the *instruct* models we prefer the
vLLM backend; this backend is the correctness-first fallback and the only one
that exposes residual-stream activations.
"""

from __future__ import annotations

import torch

from ..config import CHECKPOINTS_DIR, ModelSpec
from .base import Backend, Message


class HFBackend(Backend):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_hidden_states=False,
        )

        if spec.is_finetune:
            # Load our LoRA adapter for this finetune (Section 4).
            from peft import PeftModel

            adapter_path = CHECKPOINTS_DIR / spec.key
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            self.model = self.model.merge_and_unload()  # fold LoRA for fast inference

        self.model.eval()
        self.is_base = spec.kind == "base"

    # -- prompt construction ---------------------------------------------------
    def _build_input_ids(self, messages: list[Message], prefill: str | None = None):
        """Build input token ids.

        Instruct models use the chat template; base models (no template) get a
        plain transcript so we can prefill them consistently (Section 3.1).
        """
        if self.is_base:
            # Plain text transcript; base models continue from the assistant tag.
            text = self._plain_transcript(messages)
            if prefill is not None:
                text += prefill
            return self.tokenizer(text, return_tensors="pt").input_ids
        # Instruct: apply chat template, leaving the generation prompt open.
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if prefill:
            prefill_ids = self.tokenizer(
                prefill, return_tensors="pt", add_special_tokens=False
            ).input_ids
            ids = torch.cat([ids, prefill_ids], dim=1)
        return ids

    @staticmethod
    def _plain_transcript(messages: list[Message]) -> str:
        lines = []
        for m in messages:
            role = m["role"].capitalize()
            lines.append(f"{role}: {m['content']}")
        lines.append("Assistant:")  # open assistant turn for base-model continuation
        return "\n\n".join(lines)

    # -- generation ------------------------------------------------------------
    @torch.no_grad()
    def _sample(self, input_ids, n, max_new_tokens, temperature, top_p) -> list[str]:
        input_ids = input_ids.to(self.model.device)
        gen = self.model.generate(
            input_ids=input_ids,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        prompt_len = input_ids.shape[1]
        out = []
        for seq in gen:
            text = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            out.append(text)
        return out

    def generate(self, messages, n=1, max_new_tokens=2048, temperature=1.0, top_p=1.0):
        ids = self._build_input_ids(messages)
        return self._sample(ids, n, max_new_tokens, temperature, top_p)

    def supports_prefill(self) -> bool:
        return True

    def generate_with_prefill(self, messages, prefill, n=1, max_new_tokens=2048,
                              temperature=1.0, top_p=1.0):
        ids = self._build_input_ids(messages, prefill=prefill)
        # The decoded output already excludes the prefill (it is part of the
        # prompt prefix we slice off in _sample).
        return self._sample(ids, n, max_new_tokens, temperature, top_p)

    # -- hidden states (Appendix I) -------------------------------------------
    def supports_hidden_states(self) -> bool:
        return True

    @torch.no_grad()
    def forward_hidden_states(self, messages, prefill: str | None = None):
        """Return (token_ids, hidden_states) for a fully-specified conversation.

        `hidden_states` is a tuple of [n_layers+1] tensors, each
        (seq_len, d_model), produced by a single forward pass (no sampling).
        Used by the logit-based internal-emotion detector.
        """
        ids = self._build_input_ids(messages, prefill=prefill).to(self.model.device)
        out = self.model(ids, output_hidden_states=True)
        hidden = tuple(h[0].float().cpu() for h in out.hidden_states)
        return ids[0].cpu(), hidden

    def unembed(self, hidden_layer: "torch.Tensor"):
        """Project residual-stream activations to vocab logits via the LM head."""
        W = self.model.get_output_embeddings().weight  # (vocab, d_model)
        with torch.no_grad():
            return hidden_layer.to(W.device, W.dtype) @ W.T
