"""Local HuggingFace backend for Gemma (instruct, base, and LoRA finetunes).

Gemma is run locally because the experiments need (a) thousands of samples at
T=1, (b) response *prefilling* for the base-vs-instruct comparison, and (c) the
ability to load LoRA adapters from Section 4. The 27B model realistically needs
a sizeable GPU (or 4-bit loading via bitsandbytes); we expose dtype / quant /
device-map knobs and default to bfloat16 with device_map="auto".
"""
from __future__ import annotations

from typing import Optional

from .base import ChatMessage, GenerationResult, ModelClient


class HFModel(ModelClient):
    def __init__(
        self,
        spec,
        *,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        hf_token: str | None = None,
    ) -> None:
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        torch_dtype = getattr(torch, dtype)

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            token=hf_token,
            **quant_kwargs,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.is_base = spec.is_base

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _render_chat(self, messages: list[ChatMessage],
                     add_generation_prompt: bool) -> str:
        """Apply the chat template for instruct models.

        Gemma's template has no dedicated system role; a leading system message
        is merged into the first user turn (handled here so callers can use a
        system role uniformly).
        """
        msgs = _merge_system_into_first_user(messages)
        return self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in msgs],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _render_base(self, messages: list[ChatMessage]) -> str:
        """Plain-text rendering for base (non-chat) models.

        Base models are not trained on the chat template, so we present the
        conversation as labelled turns. In practice base models are only invoked
        via :meth:`generate_with_prefill`, where the prefix forces a consistent
        continuation (Section 3.1).
        """
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(m.content)
            elif m.role == "user":
                parts.append(f"User: {m.content}")
            else:
                parts.append(f"Assistant: {m.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _generate_from_text(
        self,
        prompt_text: str,
        temperature: float,
        max_new_tokens: int,
    ) -> GenerationResult:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt",
                                add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=bool(do_sample),
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id
                or self.tokenizer.eos_token_id,
            )
        new_ids = out[0][prompt_len:].tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenerationResult(text=text.strip(), new_token_ids=new_ids)

    def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = max_new_tokens or self.spec.max_new_tokens
        if self.is_base:
            prompt_text = self._render_base(messages)
        else:
            prompt_text = self._render_chat(messages, add_generation_prompt=True)
        return self._generate_from_text(prompt_text, temperature, max_new_tokens)

    def generate_with_prefill(
        self,
        messages: list[ChatMessage],
        prefill: str,
        *,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        temperature = self.spec.temperature if temperature is None else temperature
        max_new_tokens = max_new_tokens or self.spec.max_new_tokens
        if self.is_base:
            prompt_text = self._render_base(messages) + " " + prefill
        else:
            # Render with generation prompt, then append the forced prefix so the
            # model continues *inside* the assistant turn.
            prompt_text = self._render_chat(
                messages, add_generation_prompt=True) + prefill
        return self._generate_from_text(prompt_text, temperature, max_new_tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def _merge_system_into_first_user(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    if not messages or messages[0].role != "system":
        return messages
    system = messages[0]
    rest = messages[1:]
    for i, m in enumerate(rest):
        if m.role == "user":
            merged = ChatMessage(
                "user", f"{system.content}\n\n{m.content}")
            return rest[:i] + [merged] + rest[i + 1:]
    # No user turn -> demote system to a user turn.
    return [ChatMessage("user", system.content)] + rest
