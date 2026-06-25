"""Local Gemma client built on HuggingFace ``transformers``.

This client backs every experiment that needs weight access: the main eval for
open-weight Gemma, the prefill base-vs-instruct study (Section 3), DPO/SFT
finetuning (Section 4), and the logit-based emotion probe (Appendix I).

Two response-formatting paths:
  * instruct models -> the tokenizer chat template (``<start_of_turn>`` etc).
  * base/pretrained models -> a plain transcript, since they were never trained
    on chat formatting. The prefill study relies on this to compare the two
    fairly from identical prefilled starting points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import ChatModel, GenerationConfig, Message

if TYPE_CHECKING:  # avoid importing torch/transformers at module import time
    import torch


def _plain_transcript(messages: list[Message]) -> str:
    """Render a conversation as a plain transcript for base models."""
    parts = []
    for m in messages:
        prefix = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
        parts.append(f"{prefix}: {m.content}")
    parts.append("Assistant:")  # cue the model to continue as the assistant
    return "\n\n".join(parts)


class LocalHFModel(ChatModel):
    def __init__(
        self,
        name: str,
        hf_id: str,
        family: str = "gemma",
        is_instruct: bool = True,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        adapter_path: str | None = None,
    ):
        super().__init__(name=name, family=family, is_instruct=is_instruct)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_id = hf_id
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: dict = {
            "torch_dtype": getattr(torch, dtype),
            "device_map": device_map,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
        if adapter_path:
            # Load a trained LoRA adapter (DPO/SFT output) on top of the base.
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self._torch = torch

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------
    def render(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        """Turn messages into the model's input string."""
        if self.is_instruct:
            return self.tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        return _plain_transcript(messages)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def _generate(self, prompt_text: str, gen: GenerationConfig) -> str:
        torch = self._torch
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=gen.temperature > 0,
                temperature=max(gen.temperature, 1e-6),
                top_p=gen.top_p,
                max_new_tokens=gen.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat(self, messages: list[Message], gen: GenerationConfig) -> str:
        return self._generate(self.render(messages), gen)

    def chat_batch(
        self, conversations: list[list[Message]], gen: GenerationConfig
    ) -> list[str]:
        """Left-padded batched generation for throughput."""
        torch = self._torch
        prompts = [self.render(c) for c in conversations]
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            self.model.device
        )
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=gen.temperature > 0,
                temperature=max(gen.temperature, 1e-6),
                top_p=gen.top_p,
                max_new_tokens=gen.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_only = out[:, enc["input_ids"].shape[1]:]
        return [
            self.tokenizer.decode(row, skip_special_tokens=True).strip()
            for row in gen_only
        ]

    # ------------------------------------------------------------------
    # Prefill continuation (Section 3)
    # ------------------------------------------------------------------
    def continue_prefill(
        self, messages: list[Message], prefill: str, gen: GenerationConfig
    ) -> str:
        """Continue an assistant turn that starts with ``prefill``.

        We render the conversation with a generation prompt and append the
        prefill text *before* the generation cursor, so the model continues
        from inside the assistant turn. Only the newly sampled text is returned.
        """
        base = self.render(messages, add_generation_prompt=True)
        prompt_text = base + prefill
        continuation = self._generate(prompt_text, gen)
        return continuation

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        """Return the first ``n_tokens`` of ``text`` (the "early" truncation)."""
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)
