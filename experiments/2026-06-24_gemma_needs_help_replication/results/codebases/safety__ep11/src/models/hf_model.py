"""Local Gemma backend (HuggingFace transformers).

Handles chat-template formatting, batched sampling at temperature 1, true
prefilling for the Section 3 experiment, optional LoRA-adapter loading for the
finetuned variants, and residual-stream access for Appendix I.

The model is loaded lazily and cached so repeated ``load_model`` calls for the
same checkpoint share weights.
"""
from __future__ import annotations

import functools
from typing import Optional, Sequence

import config
from .base import ChatModel, Message


@functools.lru_cache(maxsize=4)
def _load_hf(repo_id: str, dtype: str, device_map: str):
    """Load tokenizer + model once per (repo, dtype, device_map)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=getattr(torch, dtype),
        device_map=device_map,
        output_hidden_states=False,
    )
    model.eval()
    return tok, model


class HFChatModel(ChatModel):
    def __init__(
        self,
        name: str,
        *,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        adapter_path: Optional[str] = None,
    ):
        self.name = name if adapter_path is None else f"{name}+{adapter_path}"
        self.repo_id = config.HF_MODELS[name]
        self.is_base = name.endswith("-pt")
        self.tokenizer, self.model = _load_hf(self.repo_id, dtype, device_map)
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

    # ------------------------------------------------------------------ #
    # Prompt formatting
    # ------------------------------------------------------------------ #
    def _render(self, messages: list[Message], add_generation_prompt: bool) -> str:
        """Render a conversation to a prompt string.

        Instruct models use the Gemma chat template. Base (``-pt``) models have no
        chat template, so we fall back to a minimal turn-labelled format — the
        paper only ever drives base models via prefilling, where the rendered
        prefix is mostly the assistant's partial text anyway.
        """
        msg_dicts = [m.as_dict() for m in messages]
        if not self.is_base and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        # Base model: simple, neutral formatting.
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(m.content)
            else:
                parts.append(f"{m.role.capitalize()}: {m.content}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _sample(self, prompt: str, temperature: float, max_new_tokens: int, n: int) -> list[str]:
        import torch

        # The chat template (and our base-model scaffold) already emit BOS where
        # appropriate, so disable add_special_tokens to avoid a double-BOS, which
        # noticeably degrades Gemma generations.
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=1.0,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        # Strip the prompt tokens; decode only the newly generated portion.
        gen = out[:, inputs["input_ids"].shape[1]:]
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

    def _generate(self, messages, temperature, max_new_tokens, n):
        prompt = self._render(messages, add_generation_prompt=True)
        return self._sample(prompt, temperature, max_new_tokens, n)

    def continue_from_prefill(self, messages, prefill, *, temperature=config.TEMPERATURE,
                              max_new_tokens=config.MAX_NEW_TOKENS, n=1):
        # Build the prompt up to (and including) the start of the assistant turn,
        # then append the prefill so the model continues it.
        base = self._render(list(messages), add_generation_prompt=True)
        prompt = base + prefill
        return self._sample(prompt, temperature, max_new_tokens, n)

    # ------------------------------------------------------------------ #
    # Residual-stream access for Appendix I (internal emotion detection)
    # ------------------------------------------------------------------ #
    def hidden_states_for_text(self, messages: list[Message]):
        """Return (tokens, per-layer residual stream) for a rendered conversation.

        Used by src/internal/emotion_logits.py to unembed the residual stream and
        measure emotion-token logits at each layer / position.
        """
        import torch

        prompt = self._render(messages, add_generation_prompt=False)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        # hidden_states: tuple(num_layers+1) of (1, seq, d_model)
        return inputs["input_ids"][0], out.hidden_states

    @property
    def unembed(self):
        """The output embedding / LM head weight (vocab x d_model)."""
        return self.model.get_output_embeddings().weight
