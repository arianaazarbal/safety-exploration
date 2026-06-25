"""Local HuggingFace inference for the Gemma participants (open weights).

Supports both instruct (chat-templated) and base/pretrained (raw-text)
checkpoints, and assistant-turn prefilling required by Section 3.1.

Optionally loads a PEFT/LoRA adapter on top of the base weights so the same
client serves the vanilla, DPO, and SFT Gemma models (Section 4).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from .base import ChatModel, Conversation, Message


class HFChatModel(ChatModel):
    def __init__(
        self,
        key: str,
        model_id: str,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.key = key
        self.model_id = model_id
        self.is_base = is_base

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"torch_dtype": getattr(torch, dtype), "device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        if adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()  # fold LoRA for inference speed

        self.model.eval()

    # -- prompt construction ------------------------------------------------
    def _render(self, messages: Conversation, add_generation_prompt: bool = True) -> str:
        """Render a conversation to a string the model can continue from."""
        if self.is_base:
            # Base/pretrained checkpoints are not chat-tuned. We present the
            # conversation as a transcript and let the model continue.
            # (Section 3.1: base models are compared via prefilled continuations,
            #  so this path is primarily exercised through prefill_continue.)
            lines = []
            for m in messages:
                tag = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
                    m.role, m.role.capitalize()
                )
                lines.append(f"{tag}: {m.content}")
            if add_generation_prompt:
                lines.append("Assistant:")
            return "\n".join(lines)

        return self.tokenizer.apply_chat_template(
            [m.to_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @torch.no_grad()
    def _complete(self, prompt_text: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        do_sample = temperature is not None and temperature > 0
        out = self.model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    # -- interface ----------------------------------------------------------
    def generate(
        self,
        messages: Conversation,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        stop: Optional[list[str]] = None,
    ) -> str:
        text = self._complete(self._render(messages), temperature, max_new_tokens)
        return _apply_stop(text, stop)

    def prefill_continue(
        self,
        messages: Conversation,
        prefill: str,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        # Render up to the generation prompt, then append the prefill so the
        # model continues the assistant turn from that exact point.
        base_prompt = self._render(messages, add_generation_prompt=True)
        full_prompt = base_prompt + prefill
        return self._complete(full_prompt, temperature, max_new_tokens)

    @torch.no_grad()
    def batch_generate(
        self,
        conversations: Sequence[Conversation],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        stop: Optional[list[str]] = None,
    ) -> list[str]:
        prompts = [self._render(c) for c in conversations]
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        do_sample = temperature is not None and temperature > 0
        out = self.model.generate(
            **enc,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        results = []
        for i in range(len(prompts)):
            gen = out[i, enc["input_ids"].shape[1]:]
            results.append(_apply_stop(self.tokenizer.decode(gen, skip_special_tokens=True), stop))
        return results


def _apply_stop(text: str, stop: Optional[list[str]]) -> str:
    if not stop:
        return text
    cut = len(text)
    for s in stop:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
