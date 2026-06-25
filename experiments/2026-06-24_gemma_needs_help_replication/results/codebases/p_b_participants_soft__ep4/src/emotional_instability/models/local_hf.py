"""Local open-weights client for Gemma (transformers, optional vLLM).

Responsibilities:
  * chat generation with Gemma's chat template (instruct models)
  * raw / prefilled continuation for both base ("-pt") and instruct models,
    used by the Section-3 prefill experiment
  * loading LoRA adapters produced by Section-4 training

Two engines:
  * vLLM (if installed and `EI_USE_VLLM=1`): fast batched decoding -- strongly
    recommended for the Section-2 sample counts on the 27B model.
  * transformers: always-available fallback, supports prefill continuation
    cleanly via `continue_final_message`.

Base ("-pt") models have no chat template. For them we build prompts by simple
concatenation; in practice the prefill experiment is the only place base models
are used, and it supplies an explicit assistant prefix to continue from.
"""
from __future__ import annotations

import os
from typing import Sequence

import torch

from .base import ChatMessage, GenerationConfig, ModelClient


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LocalHFClient(ModelClient):
    def __init__(
        self,
        name: str,
        hf_id: str,
        *,
        chat_template: bool = True,
        lora_adapter: str | None = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        load_in_4bit: bool = False,
    ):
        super().__init__(name)
        self.hf_id = hf_id
        self.has_chat_template = chat_template
        self.lora_adapter = lora_adapter
        self._dtype = getattr(torch, dtype)
        self._device_map = device_map
        self._load_in_4bit = load_in_4bit
        self._use_vllm = os.environ.get("EI_USE_VLLM") == "1"
        self._model = None
        self._tokenizer = None
        self._vllm = None

    # ------------------------------------------------------------------ #
    # Lazy loading: a process may construct several specs but only run one.
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._use_vllm:
            self._ensure_vllm()
        else:
            self._ensure_transformers()

    def _ensure_transformers(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.hf_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"  # left-pad for correct batched generation

        quant = None
        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._dtype,
                bnb_4bit_quant_type="nf4",
            )
        model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            torch_dtype=self._dtype,
            device_map=self._device_map,
            quantization_config=quant,
        )
        if self.lora_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.lora_adapter)
            model = model.merge_and_unload()
        model.eval()
        self._model, self._tokenizer = model, tok

    def _ensure_vllm(self) -> None:
        if self._vllm is not None:
            return
        from vllm import LLM

        kwargs: dict = {"model": self.hf_id, "dtype": "bfloat16"}
        if self.lora_adapter:
            kwargs.update(enable_lora=True)
        self._vllm = LLM(**kwargs)
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)

    # ------------------------------------------------------------------ #
    # Prompt rendering
    # ------------------------------------------------------------------ #
    def _render(
        self,
        messages: Sequence[ChatMessage],
        *,
        assistant_prefix: str | None = None,
    ) -> str:
        """Render messages to a single prompt string.

        If `assistant_prefix` is given, the prompt ends mid-assistant-turn so
        decoding continues that turn (used for prefill)."""
        msgs = [m.as_dict() for m in messages]
        if self.has_chat_template:
            if assistant_prefix is not None:
                msgs = msgs + [{"role": "assistant", "content": assistant_prefix}]
                text = self._tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    continue_final_message=True,
                )
            else:
                text = self._tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            return text
        # Base model: no chat template. Concatenate plainly. The prefill
        # experiment provides the assistant prefix that gives base models a
        # consistent starting point to continue from.
        parts = []
        for m in msgs:
            parts.append(m["content"])
        if assistant_prefix is not None:
            parts.append(assistant_prefix)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _gen_transformers(
        self, prompts: list[str], cfg: GenerationConfig, seeds: Sequence[int] | None
    ) -> list[str]:
        tok, model = self._tokenizer, self._model
        results: list[str] = []
        # Group into a single batched call when seeds are uniform; otherwise we
        # must seed per sample, so fall back to per-sample generation.
        if seeds is None:
            _seed_everything(cfg.seed)
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=cfg.temperature > 0,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1]:]
            return tok.batch_decode(gen, skip_special_tokens=True)
        for prompt, seed in zip(prompts, seeds):
            _seed_everything(seed)
            enc = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    do_sample=cfg.temperature > 0,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                )
            gen = out[0, enc["input_ids"].shape[1]:]
            results.append(tok.decode(gen, skip_special_tokens=True))
        return results

    def _gen_vllm(
        self, prompts: list[str], cfg: GenerationConfig, seeds: Sequence[int] | None
    ) -> list[str]:
        from vllm import SamplingParams

        # vLLM seeds per-request; if a single seed is given apply it uniformly.
        outs = []
        for i, prompt in enumerate(prompts):
            sp = SamplingParams(
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens,
                seed=(seeds[i] if seeds is not None else cfg.seed),
            )
            outs.append(sp)
        # vLLM.generate accepts a list of prompts but one SamplingParams; to
        # honour per-sample seeds we call once per unique seed group. For
        # simplicity and determinism, call per prompt (vLLM still batches
        # internally across the engine).
        results = []
        for prompt, sp in zip(prompts, outs):
            r = self._vllm.generate([prompt], sp)
            results.append(r[0].outputs[0].text)
        return results

    def generate(self, messages: Sequence[ChatMessage], cfg: GenerationConfig) -> str:
        return self.generate_batch([messages], cfg, seeds=None)[0]

    def generate_batch(
        self,
        batch: Sequence[Sequence[ChatMessage]],
        cfg: GenerationConfig,
        seeds: Sequence[int] | None = None,
    ) -> list[str]:
        self._ensure_loaded()
        prompts = [self._render(conv) for conv in batch]
        if self._use_vllm:
            return self._gen_vllm(prompts, cfg, seeds)
        return self._gen_transformers(prompts, cfg, seeds)

    def continue_from(
        self,
        messages: Sequence[ChatMessage],
        assistant_prefix: str,
        cfg: GenerationConfig,
    ) -> str:
        self._ensure_loaded()
        prompt = self._render(messages, assistant_prefix=assistant_prefix)
        if self._use_vllm:
            return self._gen_vllm([prompt], cfg, None)[0]
        return self._gen_transformers([prompt], cfg, None)[0]

    def continue_from_batch(
        self,
        messages: Sequence[ChatMessage],
        assistant_prefix: str,
        n: int,
        cfg: GenerationConfig,
        base_seed: int = 0,
    ) -> list[str]:
        """Generate `n` independent continuations from one prefilled prompt --
        the inner loop of the Section-3 prefill experiment (50 per prefill)."""
        self._ensure_loaded()
        prompt = self._render(messages, assistant_prefix=assistant_prefix)
        prompts = [prompt] * n
        seeds = [base_seed + i for i in range(n)]
        if self._use_vllm:
            return self._gen_vllm(prompts, cfg, seeds)
        return self._gen_transformers(prompts, cfg, seeds)
