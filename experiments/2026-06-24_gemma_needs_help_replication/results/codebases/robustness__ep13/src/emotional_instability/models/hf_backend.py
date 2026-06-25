"""Local HuggingFace backend for Gemma (instruct, base/pt, and LoRA-adapted).

Handles:
  * chat-formatted generation for instruct models (gemma-3-*-it),
  * raw prefill continuation for base models (gemma-3-*-pt), needed by the
    base-vs-instruct comparison in Section 3,
  * loading a trained LoRA adapter on top of the instruct model (Section 4
    evaluation of the DPO / SFT models).

Gemma's chat template has no dedicated system role, so a system message is
folded into the first user turn (see DESIGN.md).

This module is import-safe without torch/transformers installed; the heavy
imports happen inside ``HFBackend.__init__`` so the rest of the package can be
used (and tested) on a machine without a GPU.
"""

from __future__ import annotations

from typing import Optional

from .base import Message, ModelBackend

# Canonical HuggingFace identifiers (Appendix B.1), Gemma family only.
GEMMA_MODEL_IDS = {
    "gemma-3-27b-it": "google/gemma-3-27b-it",
    "gemma-3-27b-pt": "google/gemma-3-27b-pt",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemma-3-12b-pt": "google/gemma-3-12b-pt",
}


def _fold_system_into_first_user(messages: list[Message]) -> list[Message]:
    """Gemma chat template lacks a system role; prepend system text to the first
    user turn."""
    if not messages or messages[0].role != "system":
        return messages
    sys_text = messages[0].content
    rest = messages[1:]
    for i, m in enumerate(rest):
        if m.role == "user":
            merged = Message("user", f"{sys_text}\n\n{m.content}")
            return rest[:i] + [merged] + rest[i + 1 :]
    # No user turn: emit the system text as a user turn.
    return [Message("user", sys_text)] + rest


class HFBackend(ModelBackend):
    def __init__(
        self,
        model_id: str,
        *,
        name: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        adapter_path: Optional[str] = None,
        max_seq_len: int = 8192,
    ):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = GEMMA_MODEL_IDS.get(model_id, model_id)
        self.model_id = resolved
        self.name = name or model_id
        self.is_base = resolved.endswith("-pt")
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(resolved)

        load_kwargs: dict = dict(device_map=device_map)
        import torch as _torch

        load_kwargs["torch_dtype"] = getattr(_torch, dtype)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(resolved, **load_kwargs)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.name = name or f"{model_id}+{adapter_path}"
        self.model.eval()

    # ----------------------------------------------------------------- #
    def supports_prefill(self) -> bool:
        return True

    def _render_chat(self, messages: list[Message], add_generation_prompt: bool) -> str:
        msgs = _fold_system_into_first_user(messages)
        chat = [{"role": m.role, "content": m.content} for m in msgs]
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _generate(self, prompt_text: str, temperature: float, max_tokens: int, n: int, seed):
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        gen = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            max_new_tokens=max_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        completions = []
        for row in gen:
            new_tokens = row[prompt_len:]
            completions.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            )
        return completions

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        n: int = 1,
        stop: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> list[str]:
        if self.is_base:
            # Base models are not chat-tuned; flatten the conversation as plain
            # text. For meaningful base-model behaviour use continue_assistant().
            text = "\n\n".join(f"{m.role}: {m.content}" for m in messages) + "\n\nassistant:"
            return self._generate(text, temperature, max_tokens, n, seed)
        prompt_text = self._render_chat(messages, add_generation_prompt=True)
        return self._generate(prompt_text, temperature, max_tokens, n, seed)

    def continue_assistant(
        self,
        messages: list[Message],
        prefill: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        n: int = 1,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Continue an assistant turn beginning with `prefill`.

        For instruct models we render the chat template with a generation prompt
        and append the prefill text, so the model continues mid-turn. For base
        models we build a comparable plain-text context. The prefill is stripped
        from the returned continuations.
        """
        if self.is_base:
            context = (
                "\n\n".join(f"{m.role}: {m.content}" for m in messages)
                + f"\n\nassistant: {prefill}"
            )
        else:
            context = self._render_chat(messages, add_generation_prompt=True) + prefill
        return self._generate(context, temperature, max_tokens, n, seed)
