"""Open-weight Gemma backend (HuggingFace transformers).

Serves the instruct targets (gemma-3-27b-it, gemma-3-12b-it) and the base
checkpoint (gemma-3-27b-pt). Supports prefilling and tokenisation, which the
Section 3 prefill experiment and Section 4 finetuning both rely on.

Prompt formatting (see DESIGN.md "Base vs instruct prompt formatting"):
  * instruct models -> the official Gemma-3 chat template
  * base models     -> a plain-text transcript (base checkpoints were never
    trained on the chat template), then the assistant prefix is appended.
Both then continue from the same prefilled assistant text, which is what makes
the base/instruct comparison meaningful.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..config import Config, ModelSpec
from .base import GenerationResult, ModelClient, Turn

if TYPE_CHECKING:  # avoid importing torch at module load
    import torch


_BASE_TRANSCRIPT_TEMPLATE = (
    "The following is a conversation between a user and an AI assistant.\n\n"
)


class GemmaClient(ModelClient):
    def __init__(self, spec: ModelSpec, config: Config) -> None:
        super().__init__(spec.name, spec.model_id)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.is_base = spec.role == "base"
        token = os.environ.get("HF_TOKEN")
        self._torch = torch
        # An adapter_path in the registry loads a LoRA adapter (e.g. the DPO/SFT
        # model from Section 4) on top of the base instruct weights.
        adapter_path = spec.extra.get("adapter_path") if spec.extra else None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    # ---- prompt construction ----------------------------------------------------
    def _render_prompt(self, messages: list[Turn], assistant_prefix: str = "") -> str:
        """Render the full prompt string up to (and including) any assistant prefix."""
        if self.is_base:
            lines = [_BASE_TRANSCRIPT_TEMPLATE]
            for m in messages:
                tag = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
                lines.append(f"{tag}: {m['content']}\n")
            lines.append("Assistant: " + assistant_prefix)
            return "".join(lines)
        # Instruct: use the official chat template, leaving the assistant turn open.
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt + assistant_prefix

    # ---- generation -------------------------------------------------------------
    def _generate(self, prompt: str, *, temperature: float, max_new_tokens: int,
                  top_p: float, seed: int | None) -> tuple[str, int, int, str]:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        n_prompt = int(inputs["input_ids"].shape[1])
        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][n_prompt:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        finish = "length" if len(gen_ids) >= max_new_tokens else "stop"
        return text, n_prompt, int(gen_ids.shape[0]), finish

    def chat(self, messages, *, temperature, max_new_tokens, top_p=1.0, seed=None):
        prompt = self._render_prompt(messages)
        text, n_p, n_c, finish = self._generate(
            prompt, temperature=temperature, max_new_tokens=max_new_tokens,
            top_p=top_p, seed=seed,
        )
        return GenerationResult(text=text.strip(), prompt_tokens=n_p,
                                completion_tokens=n_c, finish_reason=finish)

    def continue_from(self, messages, prefix, *, temperature, max_new_tokens,
                      top_p=1.0, seed=None):
        prompt = self._render_prompt(messages, assistant_prefix=prefix)
        text, n_p, n_c, finish = self._generate(
            prompt, temperature=temperature, max_new_tokens=max_new_tokens,
            top_p=top_p, seed=seed,
        )
        # Return only the continuation (paper scores "excluding prefill").
        return GenerationResult(text=text, prompt_tokens=n_p,
                                completion_tokens=n_c, finish_reason=finish)

    # ---- tokenisation -----------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate_to_tokens(self, text: str, n_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def close(self) -> None:
        import gc

        del self.model
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
