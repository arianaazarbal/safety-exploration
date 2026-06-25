"""Local HuggingFace transformers client for the Gemma models.

This backend is the workhorse for everything that the closed Gemini models
cannot do: assistant-prefill continuation (Section 3 / recovery), LoRA-adapter
attachment (the finetuned variants), and residual-stream logit extraction
(Appendix I internal-emotion detection).

It is deliberately straightforward (no batching scheduler) — for large eval
sweeps prefer the vLLM backend, which produces identical samples from the same
weights. The HF backend is what you use when you need prefill or activations.
"""

from __future__ import annotations

import logging

from ..config import ModelSpec, env
from .base import ChatMessage, GenerationConfig, ModelClient

log = logging.getLogger(__name__)


class HuggingFaceClient(ModelClient):
    def __init__(self, spec: ModelSpec, device_map: str = "auto", dtype: str = "bfloat16", **kwargs):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if not spec.hf_id:
            raise ValueError(f"Model '{spec.name}' has no hf_id for the HF backend.")

        token = env("HF_TOKEN")  # gated Gemma weights need an HF token
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            token=token,
            output_hidden_states=False,
        )
        self.model.eval()

        if spec.adapter_path:
            self._attach_adapter(spec.adapter_path)

        self.is_chat = spec.chat
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------ setup
    def _attach_adapter(self, adapter_path: str) -> None:
        from peft import PeftModel

        log.info("Attaching LoRA adapter from %s", adapter_path)
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------- generation
    def _render(self, messages: list[ChatMessage], system: str | None,
                prefill: str | None = None) -> str:
        """Render a prompt string via the chat template (instruct) or by simple
        concatenation (base model)."""
        if self.is_chat:
            msgs = []
            if system:
                # Gemma's template has no separate system role; the standard
                # workaround is to prepend the system text to the first user turn.
                msgs = list(messages)
                if msgs and msgs[0].role == "user":
                    msgs = [ChatMessage("user", f"{system}\n\n{msgs[0].content}")] + msgs[1:]
                else:
                    msgs = [ChatMessage("user", system)] + msgs
            else:
                msgs = list(messages)
            template_msgs = [m.to_dict() for m in msgs]
            if prefill is not None:
                template_msgs = template_msgs + [{"role": "assistant", "content": prefill}]
                return self.tokenizer.apply_chat_template(
                    template_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            return self.tokenizer.apply_chat_template(
                template_msgs, tokenize=False, add_generation_prompt=True
            )
        # Base model: no chat formatting. Concatenate turns plainly and append
        # the prefill so the model "continues the response" (Section 3).
        parts = []
        if system:
            parts.append(system)
        for m in messages:
            parts.append(m.content)
        text = "\n".join(parts)
        if prefill is not None:
            text = text + "\n" + prefill if text else prefill
        return text

    def _sample(self, prompt: str, cfg: GenerationConfig) -> list[str]:
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            num_return_sequences=cfg.n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if cfg.temperature > 0:
            gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        # Strip the prompt; decode only the newly generated tokens.
        new_tokens = out[:, prompt_len:]
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in new_tokens]

    def generate(
        self,
        messages: list[ChatMessage],
        cfg: GenerationConfig,
        system: str | None = None,
    ) -> list[str]:
        return self._sample(self._render(messages, system), cfg)

    def continue_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        cfg: GenerationConfig,
        system: str | None = None,
    ) -> list[str]:
        prompt = self._render(messages, system, prefill=prefill)
        return self._sample(prompt, cfg)

    def supports_prefill(self) -> bool:
        return True

    # ----------------------------------------------------- activations (App. I)
    def residual_logits(self, text: str):
        """Return per-layer logit-lens distributions for ``text``.

        For each transformer layer ``L`` we take the residual-stream activations,
        apply the model's final norm + unembedding (``lm_head``), and return a
        tensor of shape ``(num_layers, seq_len, vocab)``. The internal-emotion
        detector aggregates these over emotion-related tokens (Appendix I).
        """
        torch = self._torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple: (embeddings, layer_1, ..., layer_N)
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        lm_head = base.get_output_embeddings()
        norm = self._final_norm(base)
        layer_logits = []
        for hs in hidden_states[1:]:  # skip the embedding layer
            normed = norm(hs) if norm is not None else hs
            logits = lm_head(normed)
            layer_logits.append(logits.squeeze(0).float().cpu())
        return torch.stack(layer_logits), inputs["input_ids"].squeeze(0).cpu()

    @staticmethod
    def _final_norm(model):
        """Best-effort lookup of the final RMSNorm before the unembedding."""
        for attr in ("model", "language_model"):
            inner = getattr(model, attr, None)
            if inner is not None and hasattr(inner, "norm"):
                return inner.norm
        return getattr(model, "norm", None)

    def vocab_size(self) -> int:
        return len(self.tokenizer)
