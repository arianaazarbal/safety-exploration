"""Local HuggingFace target models (Gemma instruct + base).

Supports three things the paper needs:
  * normal chat-formatted multi-turn generation (instruct models),
  * base/pretrained generation via *prefilling* (Section 3), where we force the
    model to continue a given response prefix, and
  * loading a LoRA adapter on top of gemma-3-27b-it (the SFT / DPO finetunes).

Generation is at temperature 1 by default (the paper's setting). Thinking is not
applicable to Gemma open weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .base import ChatMessage, ChatModel, Generation


@dataclass
class HFLocalModel(ChatModel):
    name: str
    model_id: str
    is_instruct: bool = True
    adapter_dir: Optional[str] = None   # LoRA adapter (SFT/DPO finetunes)
    device_map: str = "auto"
    load_in_4bit: bool = False          # 4-bit (bitsandbytes) for 27B on one GPU
    dtype: str = "bfloat16"

    _model: object = None
    _tokenizer: object = None

    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_id)
        kwargs: dict = {"device_map": self.device_map}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            import torch

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            import torch

            kwargs["torch_dtype"] = getattr(torch, self.dtype)

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_dir)
            model = model.merge_and_unload()
        model.eval()
        self._tokenizer = tok
        self._model = model

    # ------------------------------------------------------------------ #
    def _format_chat(self, messages: Sequence[ChatMessage], add_generation_prompt: bool) -> str:
        """Apply the model's chat template (instruct) or a simple concatenation
        (base models, which have no chat template)."""
        tok = self._tokenizer
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        if self.is_instruct and tok.chat_template:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Base model: emulate a minimal chat layout (see DESIGN.md — Section 3
        # prefilling relies on a consistent continuation point, not the official
        # chat format).
        parts = []
        for m in msgs:
            parts.append(f"{m.role.capitalize()}: {m.content}")
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n".join(parts)

    def _generate_from_text(
        self, prompt_text: str, temperature: float, max_new_tokens: int
    ) -> str:
        import torch

        tok = self._tokenizer
        inputs = tok(prompt_text, return_tensors="pt").to(self._model.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    def generate(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> Generation:
        self._ensure_loaded()
        prompt_text = self._format_chat(messages, add_generation_prompt=True)
        text = self._generate_from_text(prompt_text, temperature, max_new_tokens)
        return Generation(text=text.strip())

    def generate_with_prefill(
        self,
        messages: Sequence[ChatMessage],
        prefill: str,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> Generation:
        """Force the model to *continue* `prefill`. We append the prefill text
        directly after the assistant generation prompt and decode only the
        newly generated continuation."""
        self._ensure_loaded()
        prompt_text = self._format_chat(messages, add_generation_prompt=True) + prefill
        continuation = self._generate_from_text(prompt_text, temperature, max_new_tokens)
        return Generation(text=continuation.strip(), prefill=prefill)

    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> list[int]:
        self._ensure_loaded()
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def detokenize(self, ids: list[int]) -> str:
        self._ensure_loaded()
        return self._tokenizer.decode(ids, skip_special_tokens=True)
