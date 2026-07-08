"""Local Gemma backend (HuggingFace transformers).

Handles both instruct (-it) and base (-pt) checkpoints, and optionally a PEFT
LoRA adapter on top (for evaluating the DPO/SFT models). Supports:
  * batched multi-sample generation at temperature 1,
  * prefilled assistant continuations (Section 3),
  * mid-layer hidden-state extraction for the internal-emotion probe (Appendix I).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .base import ChatModel, GenerationConfig, Message


class GemmaModel(ChatModel):
    def __init__(self, spec, adapter_path: Optional[str] = None, load_4bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(self, messages: Sequence[Message], add_generation_prompt: bool) -> str:
        """Render a chat into a prompt string.

        Base (-pt) models have no chat template, so we fall back to a minimal
        Gemma-style turn format. Instruct models use the official template.
        """
        if self.spec.is_base:
            return self._render_base(messages, add_generation_prompt)
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
        )

    @staticmethod
    def _render_base(messages: Sequence[Message], add_generation_prompt: bool) -> str:
        # Plain concatenation in the Gemma turn style; base models continue from
        # this without ever having been chat-tuned (Section 3 prefill setup).
        parts = []
        for m in messages:
            tag = "user" if m["role"] != "assistant" else "model"
            parts.append(f"<start_of_turn>{tag}\n{m['content']}<end_of_turn>\n")
        if add_generation_prompt:
            parts.append("<start_of_turn>model\n")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _sample(self, prompt: str, cfg: GenerationConfig) -> list[str]:
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=1.0,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=cfg.n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def generate(self, messages: Sequence[Message], cfg: GenerationConfig) -> list[str]:
        prompt = self._render(messages, add_generation_prompt=True)
        return self._sample(prompt, cfg)

    def continue_from_prefill(
        self, messages: Sequence[Message], prefill: str, cfg: GenerationConfig
    ) -> list[str]:
        # Render up to the open assistant turn, then append the prefill text so
        # the model continues from it. Return only the part after the prefill.
        prompt = self._render(messages, add_generation_prompt=True) + prefill
        return self._sample(prompt, cfg)

    # ------------------------------------------------------------------ #
    # Internals (Appendix I logit/hidden-state probe)
    # ------------------------------------------------------------------ #
    def supports_internals(self) -> bool:
        return True

    @torch.no_grad()
    def hidden_states(
        self, messages: Sequence[Message], assistant_text: str, layers: Sequence[int]
    ) -> dict[int, torch.Tensor]:
        """Mean-pooled hidden states over the assistant span at given layers."""
        prompt = self._render(messages, add_generation_prompt=True)
        full = prompt + assistant_text
        p_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        f = self.tokenizer(full, return_tensors="pt").to(self.model.device)
        out = self.model(**f, output_hidden_states=True)
        start = p_ids.shape[1]
        result = {}
        for layer in layers:
            hs = out.hidden_states[layer][0, start:, :]  # [span, hidden]
            result[layer] = hs.mean(dim=0).float().cpu()
        return result

    def close(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
