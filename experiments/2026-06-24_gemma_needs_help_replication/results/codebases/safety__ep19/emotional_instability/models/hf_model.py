"""Local HuggingFace inference for the open-weight Gemma models.

Handles both instruct (``-it``) and pretrained/base (``-pt``) checkpoints. Base
checkpoints have no chat template, so for them we build the conversation with a
minimal role-tagged format and rely on prefilling (Section 3) to steer
continuations — exactly the regime the paper uses to compare base vs instruct.

LoRA adapters (our DPO/SFT fine-tunes) are loaded via ``adapter_path``.
"""

from __future__ import annotations

from typing import Sequence

from .base import ChatMessage, Conversation, ModelClient

# Heavy imports (torch/transformers) are deferred to __init__ so that the rest
# of the package can be imported in environments without a GPU stack.


class HFModel(ModelClient):
    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        is_base: bool = False,
        adapter_path: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        attn_implementation: str | None = "eager",  # Gemma-3 recommends eager
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.name = name or model_id.split("/")[-1]
        self.is_base = is_base
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # -- prompt construction -------------------------------------------------
    def _render(self, conversation: Conversation, prefill: str | None = None) -> str:
        msgs = [m.as_dict() if isinstance(m, ChatMessage) else m for m in conversation]
        if self.is_base or self.tokenizer.chat_template is None:
            return self._render_base(msgs, prefill)
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            text += prefill
        return text

    @staticmethod
    def _render_base(msgs: list[dict[str, str]], prefill: str | None) -> str:
        """Plain role-tagged transcript for base models (no chat template)."""
        parts = []
        for m in msgs:
            parts.append(f"{m['role'].capitalize()}: {m['content']}")
        parts.append("Assistant:")
        text = "\n\n".join(parts) + " "
        if prefill:
            text += prefill
        return text

    # -- generation ----------------------------------------------------------
    def _generate_from_text(
        self, text: str, *, temperature: float, max_tokens: int
    ) -> str:
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=1.0,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def generate(
        self,
        conversation: Conversation,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        text = self._render(conversation)
        return self._generate_from_text(
            text, temperature=temperature, max_tokens=max_tokens
        )

    def generate_with_prefill(
        self,
        conversation: Conversation,
        prefill: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> str:
        text = self._render(conversation, prefill=prefill)
        return self._generate_from_text(
            text, temperature=temperature, max_tokens=max_tokens
        )

    @property
    def supports_prefill(self) -> bool:
        return True

    # -- introspection (Appendix I) -----------------------------------------
    def residual_stream(self, conversation: Conversation, prefill: str | None = None):
        """Return (token_ids, hidden_states) where hidden_states is a tuple of
        ``(num_layers+1, seq_len, d_model)`` tensors — the per-layer residual
        stream used by the logit-lens probing in :mod:`internal_emotions`.
        """
        torch = self._torch
        text = self._render(conversation, prefill=prefill)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return inputs["input_ids"][0], out.hidden_states

    def _core(self):
        """Unwrap any PEFT wrapper to the underlying HF CausalLM."""
        m = self.model
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        return m

    def unembed(self):
        """Return the lm_head (output embedding) for the logit lens."""
        return self.model.get_output_embeddings()

    def final_norm(self):
        """Return the model's final RMSNorm (applied before the lm_head), or None.

        Handles both ``Gemma3ForCausalLM`` (``.model.norm``) and PEFT-wrapped
        variants; used by the Appendix I logit-lens probing.
        """
        core = self._core()
        inner = getattr(core, "model", core)
        return getattr(inner, "norm", None)
