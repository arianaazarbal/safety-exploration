"""Model backends.

Two implementations behind a common interface:

* :class:`HFBackend`   -- local HuggingFace inference for Gemma (instruct, base,
  and LoRA-adapted variants). Supports assistant *prefill* (needed for the
  Section 3 base-vs-instruct experiment) and raw-text continuation for base
  models.
* :class:`OpenRouterBackend` -- OpenAI-compatible API used for Gemini, with
  thinking/reasoning disabled per Appendix B.1.

All sampling uses temperature 1.0 by default (the paper's setting).
"""

from __future__ import annotations

import os
from typing import Sequence

from . import config

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #

class Backend:
    spec: config.ModelSpec

    def chat(self, messages: Sequence[Message], *, max_new_tokens: int = config.MAX_NEW_TOKENS,
             temperature: float = config.TEMPERATURE) -> str:
        """Return the assistant's reply to a chat-formatted conversation."""
        raise NotImplementedError

    def continue_assistant(self, messages: Sequence[Message], prefix: str, *,
                           max_new_tokens: int = config.MAX_NEW_TOKENS,
                           temperature: float = config.TEMPERATURE) -> str:
        """Prefill: force the assistant turn to start with ``prefix`` and return
        the *continuation only* (excluding the prefill). Used in Section 3."""
        raise NotImplementedError

    def complete(self, text: str, *, max_new_tokens: int = config.MAX_NEW_TOKENS,
                 temperature: float = config.TEMPERATURE) -> str:
        """Raw text continuation (base/pretrained models)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# HuggingFace local backend (Gemma)
# --------------------------------------------------------------------------- #

class HFBackend(Backend):
    def __init__(self, spec: config.ModelSpec, load_in_4bit: bool | None = None,
                 dtype: str = "bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.torch = torch
        if load_in_4bit is None:
            load_in_4bit = os.environ.get("EI_LOAD_4BIT", "0") == "1"

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        model_kwargs = {"device_map": "auto",
                        "torch_dtype": getattr(torch, dtype)}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_kwargs)

        # Attach a LoRA adapter if this spec is one of our finetuned variants.
        adapter = spec.extra.get("adapter")
        if adapter and os.path.isdir(adapter):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    # -- helpers ----------------------------------------------------------- #

    def _generate(self, input_ids, attention_mask, max_new_tokens, temperature):
        torch = self.torch
        gen_kwargs = dict(max_new_tokens=max_new_tokens,
                          pad_token_id=self.tokenizer.pad_token_id
                          or self.tokenizer.eos_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids,
                                       attention_mask=attention_mask, **gen_kwargs)
        new_tokens = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _encode(self, text: str):
        enc = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        return enc["input_ids"], enc["attention_mask"]

    # -- interface --------------------------------------------------------- #

    def chat(self, messages, *, max_new_tokens=config.MAX_NEW_TOKENS,
             temperature=config.TEMPERATURE):
        prompt = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True)
        input_ids, attn = self._encode(prompt)
        return self._generate(input_ids, attn, max_new_tokens, temperature)

    def continue_assistant(self, messages, prefix, *,
                           max_new_tokens=config.MAX_NEW_TOKENS,
                           temperature=config.TEMPERATURE):
        # Build the chat prompt up to the assistant generation point, then
        # append the prefill text so the model continues from it.
        prompt = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True)
        prompt = prompt + prefix
        input_ids, attn = self._encode(prompt)
        return self._generate(input_ids, attn, max_new_tokens, temperature)

    def complete(self, text, *, max_new_tokens=config.MAX_NEW_TOKENS,
                 temperature=config.TEMPERATURE):
        input_ids, attn = self._encode(text)
        return self._generate(input_ids, attn, max_new_tokens, temperature)


# --------------------------------------------------------------------------- #
# OpenRouter API backend (Gemini)
# --------------------------------------------------------------------------- #

class OpenRouterBackend(Backend):
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, spec: config.ModelSpec):
        from openai import OpenAI

        self.spec = spec
        self.client = OpenAI(base_url=self.BASE_URL,
                             api_key=config.get_key("OPENROUTER_API_KEY"))

    def chat(self, messages, *, max_new_tokens=config.MAX_NEW_TOKENS,
             temperature=config.TEMPERATURE):
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(5),
               wait=wait_exponential(multiplier=1, min=2, max=30))
        def _call():
            resp = self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_new_tokens,
                # Disable thinking/reasoning (Appendix B.1). Gemini-2.5-Pro may
                # still emit hidden reasoning that the API cannot suppress.
                extra_body={"reasoning": {"max_tokens": 0, "exclude": True}},
            )
            return resp.choices[0].message.content or ""

        return _call().strip()

    def continue_assistant(self, messages, prefix, **kw):
        # Assistant prefill is not reliably supported for Gemini via OpenRouter;
        # the Section 3 prefill experiment is Gemma-only in this replication.
        raise NotImplementedError(
            "Prefill/continuation is only implemented for the local HF backend "
            "(Section 3 uses Gemma base + instruct only).")

    def complete(self, text, **kw):
        raise NotImplementedError("No base/pretrained Gemini checkpoint exists.")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_CACHE: dict[str, Backend] = {}


def get_backend(model_name: str, **kwargs) -> Backend:
    """Instantiate (and cache) the backend for a registered model name."""
    if model_name in _CACHE:
        return _CACHE[model_name]
    spec = config.MODELS[model_name]
    if spec.backend == "hf":
        backend: Backend = HFBackend(spec, **kwargs)
    elif spec.backend == "openrouter":
        backend = OpenRouterBackend(spec)
    else:
        raise ValueError(f"Unsupported backend for sampling: {spec.backend}")
    _CACHE[model_name] = backend
    return backend
