"""Local HuggingFace backend for open-weights Gemma.

Needed for:
  * Section 3 — prefilling assistant turns on base *and* instruct models so we
    can measure how each continues from a fixed point.
  * Section 4 — inference with LoRA adapters (DPO / SFT) on top of gemma-3-27b-it.

Heavy deps (torch/transformers/peft) are imported lazily so the Section 2
Gemini path stays light. Install the commented block in requirements.txt first.
"""
from __future__ import annotations

import os

from ..config import ModelSpec
from .base import ChatModel, GenerationResult, Message


class HFLocalModel(ChatModel):
    def __init__(self, spec: ModelSpec, *, device_map: str = "auto", dtype: str = "bfloat16"):
        super().__init__(spec)
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "torch + transformers are required for the hf backend; see requirements.txt"
            ) from e
        import torch

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        if spec.adapter_path and os.path.isdir(spec.adapter_path):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, spec.adapter_path)
        self.model = model
        self.model.eval()

    def supports_prefill(self) -> bool:
        return True

    # -- prompt construction --------------------------------------------- #
    def _build_prompt(self, messages: list[Message], prefill: str | None) -> str:
        """Render the conversation to a single prompt string.

        Instruct models use the chat template; base/pretrained models get a
        plain concatenation (they were never trained on the chat format, so the
        paper prefills them and measures the continuation).
        """
        if self.spec.is_base:
            # Plain text continuation. Join turns minimally; the meaningful
            # signal in Section 3 is the prefill, which is appended last.
            chunks = [m["content"] for m in messages]
            text = "\n\n".join(chunks)
            if prefill:
                text = f"{text}\n\n{prefill}" if text else prefill
            return text

        # Instruct path: use the tokenizer's chat template.
        tmpl_messages = [dict(m) for m in messages]
        if prefill:
            # Continue the final assistant turn from `prefill`.
            tmpl_messages.append({"role": "assistant", "content": prefill})
            prompt = self.tokenizer.apply_chat_template(
                tmpl_messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                tmpl_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    # -- generation ------------------------------------------------------- #
    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float = 1.0,
        max_tokens: int = 1024,
        prefill: str | None = None,
    ) -> GenerationResult:
        torch = self._torch
        prompt = self._build_prompt(messages, prefill)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(gen_ids.shape[0]),
        )

    # -- token-level helper used by Section 3 onset labelling ------------- #
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
