"""Local Hugging Face backend for Gemma (instruct, base, and LoRA-finetuned checkpoints).

Handles three things the API backends can't:
  * Gemma's chat template (no native ``system`` role -> system text is folded into the
    first user turn, matching how Gemma 3 is served).
  * True **prefilling** for the §3 base-vs-instruct continuation experiments.
  * Exposing the residual stream / unembedded logits for the App. I emotion probe.

Loading is lazy and cached per process so the 27B weights are read once.
"""
from __future__ import annotations

from functools import cached_property

from ..config import ModelSpec
from ..utils import Message
from .base import GenerationError, ModelBackend


class HFBackend(ModelBackend):
    def __init__(self, spec: ModelSpec, *, device_map: str = "auto", dtype: str = "bfloat16",
                 adapter_path: str | None = None):
        super().__init__(spec)
        self.device_map = device_map
        self.dtype = dtype
        self.adapter_path = adapter_path

    # ---- lazy model / tokenizer ------------------------------------------------------
    @cached_property
    def _torch(self):
        import torch
        return torch

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.spec.model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return tok

    @cached_property
    def model(self):
        from transformers import AutoModelForCausalLM
        torch = self._torch
        dtype = getattr(torch, self.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id, torch_dtype=dtype, device_map=self.device_map,
            attn_implementation="eager",  # Gemma 3 recommends eager attention.
        )
        if self.adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        return model

    # ---- prompt construction ---------------------------------------------------------
    def _normalise_messages(self, messages: list[Message]) -> list[Message]:
        """Fold a leading system message into the first user turn (Gemma has no system role)."""
        if messages and messages[0]["role"] == "system":
            sys_text = messages[0]["content"]
            rest = messages[1:]
            for i, m in enumerate(rest):
                if m["role"] == "user":
                    new = [dict(x) for x in rest]
                    new[i] = {"role": "user", "content": f"{sys_text}\n\n{m['content']}"}
                    return new
            # No user turn yet: keep system text as a standalone user turn.
            return [{"role": "user", "content": sys_text}, *rest]
        return list(messages)

    def _build_inputs(self, messages: list[Message], *, prefix: str | None = None):
        """Tokenise a conversation, optionally with an assistant ``prefix`` to continue."""
        torch = self._torch
        if self.spec.is_base:
            # Base (pt) models aren't chat-tuned: render as plain text and append the
            # assistant prefix so the model continues it (this is the whole point of §3).
            text = self._render_plaintext(messages, prefix)
            enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        else:
            msgs = self._normalise_messages(messages)
            if prefix is None:
                ids = self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, return_tensors="pt",
                )
            else:
                # Append a partial assistant turn and continue it (no new generation prompt).
                msgs = [*msgs, {"role": "assistant", "content": prefix}]
                ids = self.tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=False, continue_final_message=True,
                    return_tensors="pt",
                )
            enc = {"input_ids": ids}
        return {k: v.to(self.model.device) for k, v in enc.items()}

    @staticmethod
    def _render_plaintext(messages: list[Message], prefix: str | None) -> str:
        parts = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
            parts.append(f"{tag}: {m['content']}")
        parts.append("Assistant:" + (f" {prefix}" if prefix else ""))
        return "\n".join(parts)

    # ---- generation ------------------------------------------------------------------
    def _generate(self, inputs, temperature: float, max_tokens: int) -> str:
        torch = self._torch
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = temperature > 0
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=1.0 if do_sample else None,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        except Exception as e:  # noqa: BLE001 — surface as a uniform error type.
            raise GenerationError(f"HF generate failed for {self.spec.name}: {e}") from e
        new_tokens = out[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat(self, messages, *, temperature=None, max_tokens=None) -> str:
        inputs = self._build_inputs(messages)
        return self._generate(inputs, self._temperature(temperature), self._max_tokens(max_tokens))

    def continue_from(self, messages, prefix, *, temperature=None, max_tokens=None) -> str:
        inputs = self._build_inputs(messages, prefix=prefix)
        return self._generate(inputs, self._temperature(temperature), self._max_tokens(max_tokens))

    # ---- introspection for the App. I probe -----------------------------------------
    def residual_stream(self, messages: list[Message], *, prefix: str | None = None):
        """Return (hidden_states, input_ids) for a *forward pass* over the conversation.

        hidden_states: tuple of (num_layers+1) tensors, each [seq, d_model] (batch squeezed).
        Used by the probe to unembed each layer's residual stream into vocab logits.
        """
        torch = self._torch
        inputs = self._build_inputs(messages, prefix=prefix)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = tuple(h.squeeze(0) for h in out.hidden_states)
        return hidden, inputs["input_ids"].squeeze(0)

    def unembed(self, hidden_state):
        """Project a [.., d_model] residual-stream tensor to vocab logits via the LM head.

        Applies the model's final norm first, matching how the model itself reads out logits.
        """
        torch = self._torch
        model = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        norm = model.model.norm
        lm_head = model.lm_head if hasattr(model, "lm_head") else model.get_output_embeddings()
        with torch.no_grad():
            return lm_head(norm(hidden_state))
