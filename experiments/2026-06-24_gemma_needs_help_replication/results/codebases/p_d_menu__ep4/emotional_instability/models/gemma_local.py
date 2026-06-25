"""Local Gemma inference via HuggingFace ``transformers``.

This backend supports the full set of capabilities the paper needs from an
open-weights model:

* multi-turn chat at temperature 1 (Section 2),
* assistant-turn **prefilling** and continuation (Section 3, recovery),
* loading a **LoRA adapter** on top of the instruct model (the DPO/SFT mitigations),
* exposing **hidden states / logits** for the internal-emotion probing (Appendix I).

We use ``transformers`` rather than vLLM because prefilling and hidden-state
extraction are first-class here; for the large Section-2 sampling runs a vLLM
backend would be faster, and ``VLLM_AVAILABLE`` documents that swap point.
"""

from __future__ import annotations

from typing import Optional

from ..config import GenerationConfig, ModelSpec
from .base import ChatMessage, GenerationResult, ModelClient

try:  # heavy deps are optional at import time so the module can be inspected
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False


class GemmaLocalClient(ModelClient):
    supports_prefill = True

    def __init__(
        self,
        spec: ModelSpec,
        gen: Optional[GenerationConfig] = None,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers/torch are required for local Gemma inference. "
                "Install with: pip install -r requirements.txt"
            )
        self.spec = spec
        self.gen = gen or GenerationConfig()
        self.adapter_path = adapter_path

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel  # local import: optional dependency

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _render_chat(self, messages: list[ChatMessage], add_generation_prompt: bool) -> str:
        """Render messages to a prompt string.

        For instruct models we use the tokenizer's chat template. Gemma's chat
        template has no dedicated system role, so a system message is folded into
        the first user turn (the standard Gemma convention).

        For **base** (pretrained) models there is no chat template; we emulate
        the paper's prefilling protocol (Sec 3.1) with a lightweight transcript
        format so base and instruct models continue from comparable contexts.
        """
        if self.spec.instruct:
            msgs = self._fold_system(messages)
            return self.tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in msgs],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Base model: plain transcript.
        lines = []
        for m in messages:
            prefix = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
            lines.append(f"{prefix}: {m.content}")
        if add_generation_prompt:
            lines.append("Assistant:")
        return "\n".join(lines)

    @staticmethod
    def _fold_system(messages: list[ChatMessage]) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        pending_system: Optional[str] = None
        for m in messages:
            if m.role == "system":
                pending_system = (pending_system + "\n\n" + m.content) if pending_system else m.content
                continue
            if pending_system and m.role == "user":
                out.append(ChatMessage("user", f"{pending_system}\n\n{m.content}"))
                pending_system = None
            else:
                out.append(m)
        if pending_system:  # trailing system with no following user turn
            out.append(ChatMessage("user", pending_system))
        return out

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate(self, prompt: str, max_new_tokens: int, temperature: float) -> GenerationResult:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=self.gen.top_p if do_sample else None,
                top_k=self.gen.top_k or None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        token_texts = [self.tokenizer.decode([t], skip_special_tokens=True) for t in new_ids]
        return GenerationResult(text=text, token_ids=new_ids, token_texts=token_texts)

    def chat(
        self,
        messages: list[ChatMessage],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        prompt = self._render_chat(messages, add_generation_prompt=True)
        return self._generate(
            prompt,
            max_new_tokens or self.gen.max_new_tokens,
            self.gen.temperature if temperature is None else temperature,
        )

    def continue_from_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        # Render up to (and including) the open assistant turn, then append the
        # prefill text so generation continues from it.
        prompt = self._render_chat(messages, add_generation_prompt=True) + prefill
        result = self._generate(
            prompt,
            max_new_tokens or self.gen.max_new_tokens,
            self.gen.temperature if temperature is None else temperature,
        )
        # ``result.text`` already excludes the prefill (it is part of the prompt),
        # which matches "the generated continuation (excluding prefill) is scored".
        result.meta["prefill"] = prefill
        return result

    # ------------------------------------------------------------------ #
    # Tokenisation helpers (used by truncation experiments)
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    # Internals access for Appendix I (logit-based emotion detection)
    # ------------------------------------------------------------------ #
    def residual_stream_logits(
        self,
        text: str,
        layers: Optional[list[int]] = None,
    ):
        """Return per-layer unembedded logits for every token in ``text``.

        Used by :mod:`emotional_instability.internal_emotions`. Returns a tensor
        of shape ``[n_layers, n_tokens, vocab]`` (or the subset in ``layers``) by
        applying the model's final norm + unembedding to each layer's hidden
        state (a "logit lens"). Heavy; intended for small batches.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple[n_layers+1] of [1, seq, d]
        # Reach the unembedding + final norm regardless of PEFT wrapping.
        base = getattr(self.model, "base_model", self.model)
        lm_head = self.model.get_output_embeddings()
        norm = None
        for attr in ("model",):
            inner = getattr(base, attr, None)
            if inner is not None and hasattr(inner, "norm"):
                norm = inner.norm
                break
        sel = layers if layers is not None else list(range(len(hidden_states)))
        logits = []
        with torch.no_grad():
            for layer_idx in sel:
                h = hidden_states[layer_idx][0]  # [seq, d]
                if norm is not None:
                    h = norm(h)
                logits.append(lm_head(h))  # [seq, vocab]
        return torch.stack(logits, dim=0), inputs["input_ids"][0]
