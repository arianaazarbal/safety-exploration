"""Local HuggingFace inference backend for Gemma open-weight models.

Handles three things the rest of the pipeline relies on:

1. Chat templating for instruct models (``apply_chat_template``).
2. A manual Gemma-3 template for **base/pretrained** checkpoints, which have no
   chat template of their own — needed for the Section 3 prefill comparison.
3. Prefill / continuation, used both for base models and for the Section 3
   truncation experiment.

Optional LoRA adapters (DPO/SFT finetunes, and the Appendix-I layer ablations)
are loaded on top of the base weights when registered in
``config.LORA_ADAPTERS``.
"""
from __future__ import annotations

import torch

import config
from .base import ChatMessage, GenerationConfig, ModelBackend

# Gemma-3 chat control tokens, used to hand-roll a template for base models so
# that base and instruct checkpoints receive *identical* surface text in the
# prefill experiment.
_GEMMA_BOS = "<bos>"
_TURN_START = "<start_of_turn>"
_TURN_END = "<end_of_turn>"


class HFBackend(ModelBackend):
    def __init__(self, spec):
        super().__init__(spec)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, token=config.HF_TOKEN or None)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=config.HF_TOKEN or None,
        )
        self.model.eval()

        adapter = config.LORA_ADAPTERS.get(spec.name)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render_instruct(self, messages: list[ChatMessage],
                         add_generation_prompt: bool = True,
                         prefill: str | None = None) -> str:
        """Render chat messages with the model's own chat template.

        Gemma has no system role, so any system message is folded into the
        first user turn (mirroring HF's Gemma chat template behaviour).
        """
        msgs = _fold_system_into_user(messages)
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
        if prefill is not None:
            text = text + prefill
        return text

    def _render_base(self, messages: list[ChatMessage],
                     prefill: str | None = None) -> str:
        """Hand-rolled Gemma-3 turn formatting for pretrained checkpoints."""
        msgs = _fold_system_into_user(messages)
        parts = [_GEMMA_BOS]
        for m in msgs:
            role = "model" if m["role"] == "assistant" else "user"
            parts.append(f"{_TURN_START}{role}\n{m['content']}{_TURN_END}\n")
        parts.append(f"{_TURN_START}model\n")
        if prefill is not None:
            parts.append(prefill)
        return "".join(parts)

    def _render(self, messages, prefill=None, add_generation_prompt=True):
        if self.is_base:
            return self._render_base(messages, prefill=prefill)
        return self._render_instruct(
            messages, add_generation_prompt=add_generation_prompt,
            prefill=prefill)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _sample(self, prompt_text: str, n: int, cfg: GenerationConfig) -> list[str]:
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        out = self.model.generate(
            **inputs,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        )
        gen = out[:, prompt_len:]
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip()
                for g in gen]

    def generate(self, messages, n=1, cfg=None):
        cfg = cfg or GenerationConfig()
        prompt = self._render(messages)
        return self._sample(prompt, n, cfg)

    def generate_with_prefill(self, messages, prefill, n=1, cfg=None):
        cfg = cfg or GenerationConfig()
        prompt = self._render(messages, prefill=prefill)
        # Continuations only — the prefill is part of the prompt, not the output.
        return self._sample(prompt, n, cfg)


def _fold_system_into_user(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Gemma chat format has no system turn; prepend it to the first user msg."""
    if not messages or messages[0]["role"] != "system":
        return list(messages)
    system = messages[0]["content"]
    rest = messages[1:]
    folded: list[ChatMessage] = []
    injected = False
    for m in rest:
        if not injected and m["role"] == "user":
            folded.append({"role": "user",
                           "content": f"{system}\n\n{m['content']}"})
            injected = True
        else:
            folded.append(dict(m))  # type: ignore[arg-type]
    if not injected:  # no user turn at all — keep system as a user preamble
        folded.insert(0, {"role": "user", "content": system})
    return folded
