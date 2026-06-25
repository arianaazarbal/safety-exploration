"""Local HuggingFace inference for Gemma (instruct, pretrained, and finetuned).

Used for everything that needs open weights: Section 2 rollouts for Gemma, the
Section 3 prefill continuations (base + instruct), and Appendix I activation
probing. Gemini is closed and goes through ``openrouter.py`` instead.

Loading the 27B model needs a sizeable GPU; pass ``load_in_4bit=True`` (or set
EMO_LOAD_4BIT=1) to fit it on a single 48GB card via bitsandbytes.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from .base import ChatMessage, GenerationConfig, ModelClient


def _truthy(env: str) -> bool:
    return os.environ.get(env, "").lower() in {"1", "true", "yes"}


class HFLocalClient(ModelClient):
    supports_prefill = True
    supports_activations = True

    def __init__(
        self,
        model_id: str,
        name: Optional[str] = None,
        *,
        is_base_model: bool = False,
        adapter_path: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.model_id = model_id
        self.name = name or model_id.split("/")[-1]
        self.is_base_model = is_base_model
        self.adapter_path = adapter_path
        self._load_in_4bit = _truthy("EMO_LOAD_4BIT") if load_in_4bit is None else load_in_4bit
        self._device_map = device_map
        self._dtype = dtype
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------ #
    # lazy loading (so importing the module is cheap and CPU-only safe)
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch  # noqa: WPS433 (local import keeps import-time light)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_kwargs = {}
        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, self._dtype),
            device_map=self._device_map,
            output_hidden_states=False,
            **quant_kwargs,
        )
        if self.adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    # ------------------------------------------------------------------ #
    # prompt formatting
    # ------------------------------------------------------------------ #
    def _render_prompt(
        self,
        messages: Sequence[ChatMessage],
        add_generation_prompt: bool = True,
        prefill: str = "",
    ) -> str:
        """Render a chat conversation to a prompt string.

        Instruct models use the Gemma chat template. Base (pretrained) models are
        *not* trained on the chat template, so the paper renders prior turns as
        inline text and relies on prefilling. We follow Appendix A.3 / Section 3:
        for base models we present the conversation as plain text and let the
        prefill carry the assistant turn.
        """
        tok = self.tokenizer
        if self.is_base_model:
            return self._render_base_prompt(messages) + prefill

        # Gemma has no system role; fold any system message into the first user turn.
        rendered: list[dict[str, str]] = []
        sys_text = ""
        for m in messages:
            if m.role == "system":
                sys_text += (m.content + "\n\n")
            elif m.role == "user" and sys_text:
                rendered.append({"role": "user", "content": sys_text + m.content})
                sys_text = ""
            else:
                rendered.append({"role": m.role, "content": m.content})
        if sys_text:  # trailing/standalone system text
            rendered.insert(0, {"role": "user", "content": sys_text.strip()})

        text = tok.apply_chat_template(
            rendered,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return text + prefill

    @staticmethod
    def _render_base_prompt(messages: Sequence[ChatMessage]) -> str:
        """Plain-text transcript for base models (no chat special tokens)."""
        lines = []
        for m in messages:
            if m.role == "system":
                lines.append(m.content)
            elif m.role == "user":
                lines.append(f"User: {m.content}")
            else:
                lines.append(f"Assistant: {m.content}")
        lines.append("Assistant:")
        return "\n\n".join(lines) + " "

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    def _generate(self, prompt: str, cfg: GenerationConfig, n: int) -> list[str]:
        import torch

        tok = self.tokenizer
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=not self.is_base_model)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=max(cfg.temperature, 1e-5),
            top_p=cfg.top_p,
            num_return_sequences=n,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        completions = []
        for seq in out:
            gen = seq[prompt_len:]
            completions.append(tok.decode(gen, skip_special_tokens=True).strip())
        return completions

    def chat(self, messages: Sequence[ChatMessage], cfg: GenerationConfig) -> list[str]:
        prompt = self._render_prompt(messages, add_generation_prompt=True)
        return self._generate(prompt, cfg, cfg.n)

    def prefilled_continuation(
        self,
        messages: Sequence[ChatMessage],
        prefill: str,
        cfg: GenerationConfig,
    ) -> list[str]:
        # add_generation_prompt so the assistant turn is "open", then append prefill.
        prompt = self._render_prompt(messages, add_generation_prompt=True, prefill=prefill)
        return self._generate(prompt, cfg, cfg.n)
