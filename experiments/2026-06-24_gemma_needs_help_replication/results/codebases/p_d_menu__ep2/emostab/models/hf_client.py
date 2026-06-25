"""Local HuggingFace client for Gemma subject models.

Capabilities:
  - chat()              : instruct models via the Gemma chat template
  - completion / prefill: base (pt) models and assistant-turn prefilling (Sec 3)
  - LoRA adapters       : load DPO/SFT finetunes (Section 4)
  - residual capture    : hidden states per layer for Appendix I internal detection

Gemma-3 instruct models do not natively support a `system` role in the chat
template; we follow the common convention of prepending system text to the first
user turn (documented in DESIGN.md).
"""
from __future__ import annotations

from typing import Optional

from ..config import env
from .base import ChatMessage, GenerationResult, ModelClient


class HFClient(ModelClient):
    supports_prefill = True
    supports_logits = True

    def __init__(
        self,
        spec,
        *,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        token = env("HF_TOKEN")  # Gemma weights are gated
        dtype = getattr(torch, spec.dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, token=token)
        model_kwargs = dict(torch_dtype=dtype, device_map=device_map, token=token)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype
            )
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

        self.adapter_path = adapter_path
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[ChatMessage], prefill: Optional[str]) -> str:
        """Render messages to a prompt string.

        Instruct: use the chat template (system folded into first user turn).
        Base: concatenate raw text (no template); prefill is appended directly.
        """
        if self.is_base_model:
            # Base models have no chat format; we present content plainly and let
            # the prefill (if any) seed the continuation. See Section 3.
            parts = [m.content for m in messages]
            text = "\n\n".join(parts)
            if prefill:
                text = text + "\n\n" + prefill
            return text

        # Fold system into the first user message (Gemma template has no system).
        folded: list[dict] = []
        sys_buffer = ""
        for m in messages:
            if m.role == "system":
                sys_buffer += (m.content + "\n\n")
            elif m.role == "user":
                content = (sys_buffer + m.content) if sys_buffer else m.content
                sys_buffer = ""
                folded.append({"role": "user", "content": content})
            else:
                folded.append({"role": "assistant", "content": m.content})

        prompt = self.tokenizer.apply_chat_template(
            folded, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            prompt = prompt + prefill  # continue the assistant turn
        return prompt

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        torch = self.torch
        prompt = self._render(messages, prefill)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        finish = "length" if gen_ids.shape[0] >= max_new_tokens else "stop"
        # `text` is the continuation only; prefill is reported separately so
        # Section 3 can score continuations in isolation.
        return GenerationResult(
            text=text, prefill=prefill or "", finish_reason=finish
        )

    # ------------------------------------------------------------------ #
    # Residual-stream capture (Appendix I)
    # ------------------------------------------------------------------ #
    def hidden_states(self, text: str):
        """Return per-layer hidden states for `text` (no generation).

        Shape: tuple of (num_layers+1) tensors, each [1, seq_len, hidden].
        """
        torch = self.torch
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        return out.hidden_states, inputs["input_ids"][0]

    def lm_head(self):
        """Return the unembedding matrix (for logit-lens emotion scoring)."""
        return self.model.get_output_embeddings().weight  # [vocab, hidden]
