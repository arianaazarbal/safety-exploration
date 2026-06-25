"""Local HuggingFace ``transformers`` backend for Gemma models.

Responsibilities beyond plain generation:

- **Response prefilling.** For Section 3 (base-vs-instruct continuations) and the
  Section 4 recovery experiment we must force the assistant turn to begin with a
  fixed string. For instruct models we render the chat template, append the open
  assistant turn, then concatenate the prefill tokens before generating. For
  base (pretrained) models there is no chat template, so we feed a plain-text
  rendering of the conversation followed by the prefill.

- **LoRA adapters.** A trained adapter path can be attached so the same backend
  evaluates the DPO/SFT models (Section 4) and the layer-ablation variants
  (Appendix I).

- **Internals access.** :meth:`forward_with_hidden_states` returns per-layer
  residual-stream activations and the LM head, which the Appendix I logit-based
  emotion detector consumes.

The heavy imports (``torch``, ``transformers``, ``peft``) are deferred to
construction so the rest of the package (prompts, puzzles, analysis, API
backends) imports without a GPU stack present.
"""

from __future__ import annotations

import threading
from typing import Optional

from ..config import ModelSpec
from .base import ChatMessage, GenerationResult, ModelBackend


def _render_base_prompt(messages: list[ChatMessage]) -> str:
    """Plain-text rendering of a conversation for *base* models (no chat
    template). We use a simple, explicit role-tagged format; the prefill is
    appended by the caller. This mirrors the paper's approach of prefilling base
    models so they "consistently continue the response"."""
    lines = []
    for m in messages:
        role = m["role"].capitalize()
        lines.append(f"{role}: {m['content']}")
    lines.append("Assistant:")
    return "\n".join(lines) + " "


class HFBackend(ModelBackend):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
    ):
        import torch  # noqa: F401  (validate availability early)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.name = spec.name
        self.supports_prefill = spec.supports_prefill
        self.is_base = spec.is_base
        self._lock = threading.Lock()

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        model_kwargs = {"device_map": device_map, "torch_dtype": torch_dtype}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.adapter_path = adapter_path
        self.model.eval()

    # ---- generation -------------------------------------------------------- #
    def _build_inputs(self, messages: list[ChatMessage], prefill: str):
        """Return tokenised inputs whose final tokens are the (open) assistant
        turn plus any prefill, ready for ``model.generate``."""
        if self.is_base or self.tokenizer.chat_template is None:
            text = _render_base_prompt(messages) + prefill
            return self.tokenizer(text, return_tensors="pt").to(self.model.device)
        # Instruct model: render with the chat template, leaving the assistant
        # turn open (add_generation_prompt=True), then append the prefill text.
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rendered = rendered + prefill
        return self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to(
            self.model.device
        )

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        prefill: str = "",
        seed: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> GenerationResult:
        import torch

        if prefill and not self.supports_prefill:
            raise NotImplementedError(f"{self.name} does not support prefill")

        with self._lock:
            if seed is not None:
                torch.manual_seed(seed)
            inputs = self._build_inputs(messages, prefill)
            input_len = inputs["input_ids"].shape[1]
            do_sample = temperature is not None and temperature > 0
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
            new_tokens = gen[0][input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
        return GenerationResult(
            text=text,
            prefill=prefill,
            finish_reason="stop",
            raw={"input_len": input_len},
        )

    # ---- internals (Appendix I) ------------------------------------------- #
    def forward_with_hidden_states(self, text: str):
        """Run a forward pass over ``text`` and return ``(hidden_states,
        tokenizer, model)``.

        ``hidden_states`` is a tuple of per-layer tensors (one per layer + the
        embedding output), each shaped ``[seq_len, d_model]`` (batch squeezed).
        The Appendix I detector unembeds these via the LM head and aggregates
        over emotion-related tokens.
        """
        import torch

        with self._lock, torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            out = self.model(**inputs, output_hidden_states=True)
        hidden = tuple(h[0] for h in out.hidden_states)
        return hidden, self.tokenizer, self.model

    def lm_head_unembed(self, residual):
        """Project a residual-stream tensor ``[..., d_model]`` to vocab logits
        using the model's (tied) output embedding. Used by the logit-based
        emotion detector."""
        import torch

        with torch.no_grad():
            head = self.model.get_output_embeddings()
            return head(residual)

    def close(self) -> None:  # pragma: no cover
        try:
            import torch

            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
