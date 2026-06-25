"""Gemma provider backed by HuggingFace transformers.

Handles three cases the paper needs:
  * instruct chat generation (Section 2),
  * prefilled continuation for both instruct and base ("-pt") checkpoints
    (Section 3), where base models have no chat template so we continue raw text,
  * loading a LoRA adapter on top of the instruct model to evaluate the SFT/DPO
    interventions (Section 4).
"""
from __future__ import annotations

import os
from typing import Optional

from ..config import ModelSpec, SamplingConfig
from .base import ChatMessage, ModelProvider

# Heavy deps are imported lazily so that the lightweight stages (judge, analysis,
# Gemini) can run without torch/transformers installed.


class GemmaProvider(ModelProvider):
    def __init__(
        self,
        spec: ModelSpec,
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)

        model_kwargs: dict = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["torch_dtype"] = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **model_kwargs)

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.adapter_path = adapter_path
        else:
            self.adapter_path = None

        self.model.eval()

    # ------------------------------------------------------------------ #
    # prompt construction
    # ------------------------------------------------------------------ #
    def _render_chat(self, messages: list[ChatMessage], add_generation_prompt: bool) -> str:
        """Render instruct chat with Gemma's template.

        Gemma's chat template has no dedicated system role; we fold any system
        message into the first user turn (the conventional Gemma-3 approach).
        """
        rendered: list[dict] = []
        system_prefix = ""
        for m in messages:
            if m.role == "system":
                system_prefix += m.content.strip() + "\n\n"
            elif m.role == "user":
                content = (system_prefix + m.content) if system_prefix else m.content
                system_prefix = ""
                rendered.append({"role": "user", "content": content})
            else:
                rendered.append({"role": "assistant", "content": m.content})
        return self.tokenizer.apply_chat_template(
            rendered, tokenize=False, add_generation_prompt=add_generation_prompt)

    def _render_base(self, messages: list[ChatMessage]) -> str:
        """Concatenate raw text for base ("-pt") models (no chat structure)."""
        parts = [m.content for m in messages if m.content]
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    def _generate_from_text(self, prompt_text: str, sampling: SamplingConfig,
                            seed: Optional[int]) -> str:
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                max_new_tokens=sampling.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen_ids = out[0][input_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def generate(self, messages: list[ChatMessage], sampling: SamplingConfig,
                 seed: Optional[int] = None) -> str:
        if self.spec.is_base:
            # Base models have no chat surface; in Section 2 they are unused, but
            # we still support a raw continuation for completeness.
            return self._generate_from_text(self._render_base(messages), sampling, seed)
        prompt = self._render_chat(messages, add_generation_prompt=True)
        return self._generate_from_text(prompt, sampling, seed)

    def continue_from(self, messages: list[ChatMessage], prefill: str,
                      sampling: SamplingConfig, seed: Optional[int] = None) -> str:
        if self.spec.is_base:
            prompt = self._render_base(messages)
            prompt = (prompt + "\n" + prefill) if prompt else prefill
        else:
            # add_generation_prompt opens the model turn; append the prefill so
            # the model continues mid-response.
            prompt = self._render_chat(messages, add_generation_prompt=True) + prefill
        return self._generate_from_text(prompt, sampling, seed)
