"""Model backends.

Two backend kinds, behind a common interface:

* ``HFBackend``         - local HuggingFace transformers. Used for all Gemma
  models, including the base (``-pt``) checkpoints needed for the Section 3
  prefill experiment. Supports response *prefilling*, which the API backend
  cannot.
* ``OpenRouterBackend`` - OpenAI-compatible HTTP client. Used for Gemini and for
  the Claude judge / Petri agents.

The interface is deliberately small:

    backend.chat(messages, temperature, max_new_tokens) -> str
    backend.chat_prefill(messages, prefill, ...)        -> str   (HF only)
    backend.chat_batch([messages, ...], ...)            -> [str] (efficient HF)

All generation defaults to temperature 1.0 (the paper's setting).
"""

from __future__ import annotations

import abc
import os
import time
from typing import Optional, Sequence

from .. import config
from .types import Message


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class ChatBackend(abc.ABC):
    def __init__(self, spec: "config.ModelSpec"):
        self.spec = spec

    @abc.abstractmethod
    def chat(self, messages: Sequence[Message], *, temperature: float = config.TEMPERATURE,
             max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        ...

    def chat_batch(self, batch: Sequence[Sequence[Message]], *,
                   temperature: float = config.TEMPERATURE,
                   max_new_tokens: int = config.MAX_NEW_TOKENS) -> list[str]:
        # Default: sequential fallback. HFBackend overrides with true batching.
        return [self.chat(m, temperature=temperature, max_new_tokens=max_new_tokens)
                for m in batch]

    def chat_prefill(self, messages: Sequence[Message], prefill: str, *,
                     temperature: float = config.TEMPERATURE,
                     max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
        raise NotImplementedError(
            f"{type(self).__name__} does not support response prefilling "
            "(needed only for the Section 3 base/instruct experiment, which is "
            "Gemma-only and therefore uses the HF backend)."
        )


# --------------------------------------------------------------------------- #
# Local HuggingFace backend
# --------------------------------------------------------------------------- #
class HFBackend(ChatBackend):
    """Local transformers backend.

    Handles both instruct and base Gemma models. For base (``-pt``) models there
    is no chat template, so we render the conversation as plain text (see
    ``_render_base``). This only matters for the prefill experiment where base
    models are always given a prefilled assistant turn to continue.
    """

    def __init__(self, spec: "config.ModelSpec", *, device_map: str = "auto",
                 load_in_4bit: bool = False, adapter_path: Optional[str] = None):
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        dtype = getattr(torch, spec.dtype)
        # Tokenizer / chat template come from the (instruct) base, even when a
        # finetuned LoRA adapter is layered on top.
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(torch_dtype=dtype, device_map=device_map)
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        self.model = AutoModelForCausalLM.from_pretrained(spec.model_id, **load_kwargs)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.is_base = spec.kind == "base"

    # -- prompt rendering ---------------------------------------------------- #
    def _render_instruct(self, messages: Sequence[Message], add_generation_prompt: bool,
                         prefill: str | None = None) -> str:
        msgs = [m.to_dict() for m in messages]
        if prefill is not None:
            # Append a partial assistant turn and ask the template to continue it.
            msgs = msgs + [{"role": "assistant", "content": prefill}]
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                continue_final_message=True,
            )
            return text
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt,
        )

    def _render_base(self, messages: Sequence[Message], prefill: str | None = None) -> str:
        """Plain-text rendering for base models (no chat template).

        We use simple, neutral role markers. The base model is always given a
        prefilled assistant turn to continue, so the trailing text is the start
        of the assistant response (plus the prefill if provided).
        """
        parts = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m.role]
            parts.append(f"{tag}: {m.content}")
        head = "\n\n".join(parts)
        assistant_start = "\n\nAssistant:"
        if prefill:
            assistant_start += " " + prefill
        return head + assistant_start

    def _render(self, messages, *, prefill=None, add_generation_prompt=True) -> str:
        if self.is_base:
            return self._render_base(messages, prefill=prefill)
        return self._render_instruct(messages, add_generation_prompt, prefill=prefill)

    # -- generation ---------------------------------------------------------- #
    def _generate(self, prompts: list[str], temperature: float, max_new_tokens: int) -> list[str]:
        torch = self.torch
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             add_special_tokens=False).to(self.model.device)
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0 if do_sample else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

    def chat(self, messages, *, temperature=config.TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS):
        prompt = self._render(messages, add_generation_prompt=True)
        return self._generate([prompt], temperature, max_new_tokens)[0]

    def chat_batch(self, batch, *, temperature=config.TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS):
        prompts = [self._render(m, add_generation_prompt=True) for m in batch]
        return self._generate(prompts, temperature, max_new_tokens)

    def chat_prefill(self, messages, prefill, *, temperature=config.TEMPERATURE,
                     max_new_tokens=config.MAX_NEW_TOKENS):
        prompt = self._render(messages, prefill=prefill, add_generation_prompt=False)
        return self._generate([prompt], temperature, max_new_tokens)[0]

    def chat_prefill_batch(self, batch, prefill, *, temperature=config.TEMPERATURE,
                           max_new_tokens=config.MAX_NEW_TOKENS) -> list[str]:
        prompts = [self._render(m, prefill=prefill, add_generation_prompt=False) for m in batch]
        return self._generate(prompts, temperature, max_new_tokens)


# --------------------------------------------------------------------------- #
# OpenRouter (OpenAI-compatible) backend
# --------------------------------------------------------------------------- #
class OpenRouterBackend(ChatBackend):
    """OpenAI-compatible client pointed at OpenRouter (Gemini + Claude judge)."""

    def __init__(self, spec: "config.ModelSpec", *, max_retries: int = 5):
        super().__init__(spec)
        from openai import OpenAI
        self.client = OpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=config.require_env(config.OPENROUTER_API_KEY_ENV),
        )
        self.max_retries = max_retries

    def _extra_body(self) -> dict:
        # Disable provider-side reasoning/thinking where supported (paper sets
        # thinking=False for all API models; Appendix B.1).
        if self.spec.disable_thinking:
            return {"reasoning": {"enabled": False}}
        return {}

    def chat(self, messages, *, temperature=config.TEMPERATURE, max_new_tokens=config.MAX_NEW_TOKENS):
        payload = [m.to_dict() for m in messages]
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=payload,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    extra_body=self._extra_body(),
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - retry transient API errors
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenRouter call failed after {self.max_retries} retries") from last_err


# --------------------------------------------------------------------------- #
# Factory + cache
# --------------------------------------------------------------------------- #
_BACKEND_CACHE: dict[str, ChatBackend] = {}


def get_backend(model: str | "config.ModelSpec", **kwargs) -> ChatBackend:
    spec = model if isinstance(model, config.ModelSpec) else config.MODELS[model]
    key = spec.name
    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]
    if spec.backend == "hf":
        backend: ChatBackend = HFBackend(spec, **kwargs)
    elif spec.backend == "openrouter":
        backend = OpenRouterBackend(spec, **kwargs)
    else:
        raise ValueError(f"Unknown backend kind: {spec.backend!r}")
    _BACKEND_CACHE[key] = backend
    return backend


def get_finetuned_backend(base_model: str, adapter_path: str, *,
                          name: Optional[str] = None) -> HFBackend:
    """Load a base HF model with a LoRA adapter on top (for evaluating SFT/DPO
    checkpoints with the Section 2 harness). Registered in the cache under
    `name` (default: derived from the adapter path)."""
    base_spec = config.MODELS[base_model]
    name = name or f"{base_model}+{adapter_path.rstrip('/').split('/')[-1]}"
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]  # type: ignore[return-value]
    spec = config.ModelSpec(
        name=name, backend="hf", model_id=base_spec.model_id,
        family=base_spec.family, kind="instruct", dtype=base_spec.dtype,
    )
    backend = HFBackend(spec, adapter_path=adapter_path)
    _BACKEND_CACHE[name] = backend
    return backend


def get_api_backend(model_id: str, *, disable_thinking: bool = False,
                    family: str = "judge") -> OpenRouterBackend:
    """Build an ad-hoc OpenRouter backend for an arbitrary model id (judge,
    auditor, secondary judge, etc.) that is not in the MODELS registry."""
    spec = config.ModelSpec(
        name=model_id, backend="openrouter", model_id=model_id,
        family=family, kind="instruct", disable_thinking=disable_thinking,
    )
    return OpenRouterBackend(spec)
