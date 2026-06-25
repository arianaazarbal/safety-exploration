"""Local HuggingFace Gemma client.

Gemma participants run locally because three core experiments need raw weight
access that an API cannot give:
  * Section 3 prefilling (seed an assistant turn, let the model continue);
  * Section 4 DPO/SFT LoRA finetuning;
  * Appendix I activation/logit probing.

This wraps a single loaded model + tokenizer and exposes:
  * ``chat`` -- standard multi-turn generation (with optional prefill);
  * ``complete`` -- raw text completion (for base/"pt" models that have no chat
    template);
  * the underlying ``model``/``tokenizer`` for training and probing code to reuse.

Gemma-3's chat template does not take a separate ``system`` role, so we fold any
system message into the first user turn (the convention Gemma post-training used).
"""
from __future__ import annotations

from typing import Optional

from .base import ChatModel, Message


class HFGemmaModel(ChatModel):
    def __init__(
        self,
        name: str,
        hf_id: str,
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        is_base: bool = False,
        adapter_path: Optional[str] = None,
        device_map: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.hf_id = hf_id
        self.is_base = is_base  # "pt" pretrained checkpoints have no chat template

        quant_kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
            **quant_kwargs,
        )
        if adapter_path:  # load a trained LoRA adapter (DPO/SFT eval)
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # -- public API ---------------------------------------------------------
    def supports_prefill(self) -> bool:
        return True

    def chat(
        self,
        messages: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
    ) -> str:
        prompt = self._render(messages, prefill=prefill)
        return self.complete(prompt, temperature=temperature, max_new_tokens=max_new_tokens)

    def complete(self, prompt: str, temperature: float = 1.0, max_new_tokens: int = 2048) -> str:
        """Raw completion from a fully-rendered prompt string."""
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0)
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=1.0)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text

    # -- prompt rendering ---------------------------------------------------
    def _render(self, messages: list[Message], prefill: Optional[str] = None) -> str:
        """Render messages to a Gemma prompt string ending at the assistant turn.

        For instruct models we use the chat template. For base ("pt") models we
        fall back to a plain transcript -- they were never trained on the chat
        format, so Section 3 relies on *prefill* to coax a usable continuation.
        """
        if self.is_base:
            return self._render_base(messages, prefill)

        msgs = self._fold_system(messages)
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        if prefill:
            text += prefill  # seed the assistant turn; model continues from here
        return text

    @staticmethod
    def _fold_system(messages: list[Message]) -> list[Message]:
        """Gemma has no system role: prepend system text to the first user turn."""
        sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if sys_txt and rest and rest[0]["role"] == "user":
            rest = [{"role": "user", "content": f"{sys_txt}\n\n{rest[0]['content']}"}] + rest[1:]
        return rest

    @staticmethod
    def _render_base(messages: list[Message], prefill: Optional[str]) -> str:
        lines = []
        for m in messages:
            tag = {"user": "User", "assistant": "Assistant", "system": "System"}.get(m["role"], m["role"])
            lines.append(f"{tag}: {m['content']}")
        prompt = "\n".join(lines) + "\nAssistant:"
        if prefill:
            prompt += " " + prefill
        return prompt
