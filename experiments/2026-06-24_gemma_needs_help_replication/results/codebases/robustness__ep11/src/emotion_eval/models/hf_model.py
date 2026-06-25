"""Local Hugging Face model client for the Gemma family.

Handles three roles:
  - instruct models (gemma-3-*-it)  : chat via the tokenizer chat template.
  - base/pretrained models (gemma-3-*-pt) : no chat template; we build a plain-text
    transcript and rely on prefill continuation (Section 3).
  - finetuned variants : a base instruct model + a PEFT LoRA adapter directory.

Prefill is implemented natively: we render the prompt, append the (paraphrased) prefill
to the start of the assistant turn, generate, and return ONLY the continuation. This is
exactly what Sections 3 and 4 (recovery) require.

Inference defaults to plain transformers ``generate``. If vLLM is installed and
``use_vllm=True`` it can be swapped in for throughput, but transformers keeps the prefill
semantics simple and provider-independent, so it is the default.
"""
from __future__ import annotations

from typing import Sequence

from .base import ChatMessage, ModelClient


class HFModel(ModelClient):
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        is_instruct: bool = True,
        adapter_path: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.is_instruct = is_instruct
        self.adapter_path = adapter_path

        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- prompt rendering ----------------------------------------------------------------

    def _render_chat(self, messages: Sequence[ChatMessage], add_generation_prompt: bool) -> str:
        """Render a chat-formatted prompt. Instruct models use the chat template.

        Gemma chat template does not support a separate system role; we fold any system
        message into the first user turn (matching common Gemma usage).
        """
        msgs = self._fold_system(messages)
        return self.tokenizer.apply_chat_template(
            [m.as_dict() for m in msgs],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def _fold_system(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        out: list[ChatMessage] = []
        pending_system: str | None = None
        for m in messages:
            if m.role == "system":
                pending_system = m.content if pending_system is None else pending_system + "\n\n" + m.content
                continue
            if pending_system is not None and m.role == "user":
                out.append(ChatMessage("user", pending_system + "\n\n" + m.content))
                pending_system = None
            else:
                out.append(m)
        if pending_system is not None:  # system with no following user turn
            out.append(ChatMessage("user", pending_system))
        return out

    def _render_base(self, messages: Sequence[ChatMessage]) -> str:
        """Plain-text transcript for base models (no chat special tokens)."""
        lines = []
        for m in self._fold_system(messages):
            tag = "User" if m.role == "user" else "Assistant"
            lines.append(f"{tag}: {m.content}")
        lines.append("Assistant:")
        return "\n".join(lines)

    # -- generation ----------------------------------------------------------------------

    def _generate(self, prompt_text: str, *, temperature: float, max_new_tokens: int) -> str:
        import torch

        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def chat(self, messages: Sequence[ChatMessage], *, temperature: float, max_new_tokens: int) -> str:
        if self.is_instruct:
            prompt_text = self._render_chat(messages, add_generation_prompt=True)
        else:
            prompt_text = self._render_base(messages)
        return self._generate(prompt_text, temperature=temperature, max_new_tokens=max_new_tokens).strip()

    def continue_text(
        self, messages: Sequence[ChatMessage], prefill: str, *, temperature: float, max_new_tokens: int
    ) -> str:
        if self.is_instruct:
            base_prompt = self._render_chat(messages, add_generation_prompt=True)
        else:
            base_prompt = self._render_base(messages)
        prompt_text = base_prompt + prefill
        continuation = self._generate(prompt_text, temperature=temperature, max_new_tokens=max_new_tokens)
        return continuation  # excludes the prefill, per the Section 3 protocol
