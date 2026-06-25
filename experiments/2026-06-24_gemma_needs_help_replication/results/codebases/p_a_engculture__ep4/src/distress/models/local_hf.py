"""HuggingFace ``transformers`` provider for local Gemma checkpoints.

This backend is always available (no vLLM dependency) and is the only one that
supports the two things the paper's mechanistic experiments need:

* **Prefill** — continue a partially written assistant turn (Section 3).
* **Hidden states** — expose residual-stream activations (Appendix I probing).

For large throughput sweeps prefer the vLLM provider; this one is correctness-
first, not speed-first.
"""

from __future__ import annotations

from typing import Sequence

from ..config import ModelSpec
from .base import GenConfig, GenResult, Message, ModelProvider


class HFProvider(ModelProvider):
    def __init__(
        self,
        spec: ModelSpec,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
    ):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {"torch_dtype": getattr(torch, dtype), "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)
        if adapter_path:
            # Load a trained LoRA adapter (DPO/SFT output) on top of the base weights.
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.adapter_path = adapter_path

    # --- prompt construction --------------------------------------------- #
    def _render_prompt(self, messages: list[Message], prefill: str | None) -> str:
        """Return the full prompt string the model should continue from."""
        if self.spec.is_base:
            # Base (pretrained) checkpoints have no chat template. We render a
            # lightweight transcript and rely on prefill to anchor continuation,
            # exactly as the prefill experiment intends. # CHOICE (format below)
            return self._render_plaintext(messages, prefill)

        # Instruct model: use the official chat template with a generation prompt,
        # then append the prefill so the model continues the assistant turn.
        text = self.tokenizer.apply_chat_template(
            [m.to_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        if prefill:
            text += prefill
        return text

    @staticmethod
    def _render_plaintext(messages: list[Message], prefill: str | None) -> str:
        lines = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
            lines.append(f"{tag}: {m.content}")
        lines.append("Assistant:" + ((" " + prefill) if prefill else ""))
        return "\n".join(lines)

    # --- generation ------------------------------------------------------- #
    def _generate(
        self, messages: list[Message], gen: GenConfig, prefill: str | None
    ) -> GenResult:
        torch = self.torch
        prompt = self._render_prompt(messages, prefill)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        if gen.seed is not None:
            torch.manual_seed(gen.seed + gen.sample_index)

        do_sample = gen.temperature and gen.temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=gen.temperature if do_sample else None,
                top_p=gen.top_p if do_sample else None,
                max_new_tokens=gen.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0][prompt_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        text = self._strip_stops(text, gen.stop)
        return GenResult(text=text, meta={"n_new_tokens": int(new_tokens.shape[0])})

    @staticmethod
    def _strip_stops(text: str, stop: Sequence[str] | None) -> str:
        if not stop:
            return text
        cut = len(text)
        for s in stop:
            idx = text.find(s)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]

    # --- probing support (Appendix I) ------------------------------------ #
    def token_ids(self, messages: list[Message], prefill: str | None = None):
        """Tokenise a rendered prompt (used by the logit-emotion probe)."""
        prompt = self._render_prompt(messages, prefill)
        return self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    def hidden_states(self, input_ids):
        """Return per-layer residual-stream hidden states for ``input_ids``.

        Shape: tuple of (n_layers+1) tensors, each [batch, seq, d_model].
        """
        with self.torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True)
        return out.hidden_states

    def _norm_and_head_weight(self):
        """Locate the final RMSNorm and the output-embedding weight.

        Robust to (a) PEFT wrapping and (b) Gemma 3's multimodal layout, where the
        text decoder lives under ``model.language_model`` rather than ``model``.
        """
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        # Output head works uniformly across architectures.
        head = base.get_output_embeddings()
        # Find the final norm by trying known layouts in order.
        norm = None
        for path in ("model.norm", "model.language_model.norm", "language_model.model.norm"):
            obj = base
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                norm = obj
                break
            except AttributeError:
                continue
        if norm is None:
            raise AttributeError("Could not locate the model's final norm layer for unembedding.")
        return norm, head.weight  # weight: [vocab, d_model]

    def unembed(self, hidden):
        """Project a residual-stream tensor to full vocab logits via the LM head.

        Applies the model's final norm first, matching how the model actually
        reads out the residual stream. ``hidden`` is [..., d_model].
        """
        norm, weight = self._norm_and_head_weight()
        with self.torch.no_grad():
            return self.torch.matmul(norm(hidden), weight.t())

    def selective_logits(self, hidden, token_ids):
        """Logits for only ``token_ids`` (memory-safe over many positions).

        ``hidden`` is [n, d_model]; returns [n, len(token_ids)]. Used by the
        logit-emotion probe, which only needs a few thousand emotion/random tokens
        rather than the full ~256k-token vocabulary.
        """
        torch = self.torch
        norm, weight = self._norm_and_head_weight()
        idx = torch.as_tensor(list(token_ids), device=weight.device)
        with torch.no_grad():
            normed = norm(hidden.to(weight.dtype))
            return torch.matmul(normed, weight[idx].t())
