"""Gemma participant (open weights) via HuggingFace transformers.

Gemma is run locally because Sections 3 and 4 need capabilities the hosted APIs
don't expose: response *prefilling* / raw-text continuation (base vs instruct
comparison) and LoRA *finetuning*. This client therefore implements both
:class:`~..base.Participant` and :class:`~..base.Prefillable`.

A finetuned model is the same class with a LoRA adapter applied -- see
``adapter_path``. That keeps the "DPO Gemma (ours)" participant a drop-in for the
Section 2 eval suite.
"""

from __future__ import annotations

from .base import Conversation, Message


class GemmaParticipant:
    def __init__(
        self,
        name: str,
        model_id: str,
        *,
        is_base: bool = False,
        adapter_path: str | None = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.is_base = is_base
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        quant = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            quantization_config=quant,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- prompt construction ---------------------------------------------- #
    def _render_chat(self, conversation: Conversation, add_generation_prompt: bool) -> str:
        """Render a conversation to a string using the Gemma chat template.

        Gemma's template has no system role, so a leading system message is folded
        into the first user turn (the standard transformers behaviour for Gemma).
        """
        msgs = []
        system_text = None
        for m in conversation:
            if m.role == "system":
                system_text = (system_text + "\n" + m.content) if system_text else m.content
                continue
            content = m.content
            if system_text and m.role == "user" and not any(x["role"] == "user" for x in msgs):
                content = f"{system_text}\n\n{content}"
            msgs.append({"role": m.role, "content": content})
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def _generate_from_text(self, prompt_text: str, *, temperature: float, max_new_tokens: int) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # -- Participant ------------------------------------------------------- #
    def generate(self, conversation: Conversation, *, temperature: float, max_new_tokens: int) -> str:
        if self.is_base:
            # Base models have no chat format; emulate a minimal turn-structured
            # transcript so the comparison in Section 3 starts from the same place.
            prompt = _plain_transcript(conversation) + "\nAssistant:"
            return self._generate_from_text(prompt, temperature=temperature, max_new_tokens=max_new_tokens).strip()
        prompt = self._render_chat(conversation, add_generation_prompt=True)
        return self._generate_from_text(prompt, temperature=temperature, max_new_tokens=max_new_tokens).strip()

    # -- Prefillable ------------------------------------------------------- #
    def continue_response(
        self, conversation: Conversation, prefill: str, *, temperature: float, max_new_tokens: int
    ) -> str:
        if self.is_base:
            base_prompt = _plain_transcript(conversation) + "\nAssistant: " + prefill
            return self._generate_from_text(
                base_prompt, temperature=temperature, max_new_tokens=max_new_tokens
            )
        prompt = self._render_chat(conversation, add_generation_prompt=True) + prefill
        return self._generate_from_text(prompt, temperature=temperature, max_new_tokens=max_new_tokens)

    def continue_raw_text(self, text: str, *, temperature: float, max_new_tokens: int) -> str:
        return self._generate_from_text(text, temperature=temperature, max_new_tokens=max_new_tokens)

    # -- token utilities (used by the prefill module) --------------------- #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def truncate_tokens_before_end(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        keep = max(0, len(ids) - n_tokens)
        return self.tokenizer.decode(ids[:keep], skip_special_tokens=True)


def _plain_transcript(conversation: Conversation) -> str:
    """A neutral 'User:/Assistant:' transcript for base-model continuation."""
    lines = []
    for m in conversation:
        if m.role == "system":
            lines.append(m.content)
        elif m.role == "user":
            lines.append(f"User: {m.content}")
        else:
            lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)
