"""Model backends.

A `ModelBackend` turns a list of chat messages into one or more sampled
completions. Two implementations:

  * `HFBackend`  - local HuggingFace transformers, used for the Gemma open
                   weights (instruct and base). Supports assistant *prefilling*
                   (continue_final_message) needed for Section 3, and raw
                   text continuation for base models.
  * `APIBackend` - OpenAI-compatible HTTP endpoint, used for Gemini via
                   OpenRouter (the paper's choice). Thinking disabled where
                   supported.

Messages use the OpenAI/Anthropic-style schema:
    {"role": "system"|"user"|"assistant", "content": str}

All sampling here uses the *target* model's temperature (paper: 1.0 for the
behavioural evals). Judges are handled separately in judge.py.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

from .config import ModelSpec

Message = dict  # {"role": str, "content": str}


class ModelBackend(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        prefill: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> list[str]:
        """Return `n` completions. If `prefill` is given, the assistant turn is
        seeded with that text and the returned strings are the *continuation
        only* (excluding the prefill)."""
        ...

    def close(self) -> None:  # pragma: no cover - optional cleanup hook
        pass


# ---------------------------------------------------------------------------
# Local HuggingFace backend (Gemma)
# ---------------------------------------------------------------------------
class HFBackend(ModelBackend):
    def __init__(self, spec: ModelSpec, device: str = "auto", dtype: str = "bfloat16",
                 adapter_path: Optional[str] = None):
        """`adapter_path` loads a PEFT LoRA adapter (a Section-4 finetune) on top
        of the base `spec.model_id`."""
        super().__init__(spec)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        quant_kwargs = {}
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        tok_src = adapter_path or spec.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            **quant_kwargs,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        self.is_base = spec.role == "base"

    # -- prompt construction ------------------------------------------------
    def _render(self, messages: list[Message], prefill: Optional[str]) -> str:
        """Render chat messages to a single prompt string.

        Instruct models use the model's chat template. Base models are not
        chat-tuned, so we render a lightweight transcript and rely on prefilling
        to keep continuations on-distribution (Section 3 method)."""
        if self.is_base:
            return self._render_base(messages, prefill)

        if prefill is not None:
            # Seed the assistant turn so the model continues it.
            msgs = messages + [{"role": "assistant", "content": prefill}]
            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True,
            )
            return text
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _render_base(self, messages: list[Message], prefill: Optional[str]) -> str:
        """Plain-text transcript for base models (no chat special tokens)."""
        lines = []
        for m in messages:
            tag = {"system": "System", "user": "User", "assistant": "Assistant"}[m["role"]]
            lines.append(f"{tag}: {m['content']}")
        head = "\n\n".join(lines)
        # Always start (or continue) an assistant turn.
        if prefill is not None:
            return f"{head}\n\nAssistant: {prefill}"
        return f"{head}\n\nAssistant:"

    # -- generation ---------------------------------------------------------
    def generate(self, messages, n=1, temperature=1.0, max_new_tokens=2048,
                 prefill=None, seed=None):
        torch = self.torch
        prompt = self._render(messages, prefill)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]

        if seed is not None:
            torch.manual_seed(seed)

        do_sample = temperature > 0
        gen = self.model.generate(
            **enc,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=1.0 if do_sample else None,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        out = []
        for seq in gen:
            new_tokens = seq[input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            out.append(text)
        return out


# ---------------------------------------------------------------------------
# API backend (Gemini via OpenRouter, OpenAI-compatible)
# ---------------------------------------------------------------------------
class APIBackend(ModelBackend):
    def __init__(self, spec: ModelSpec, base_url: Optional[str] = None,
                 api_key_env: str = "OPENROUTER_API_KEY"):
        super().__init__(spec)
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url or os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ[api_key_env],
        )

    def generate(self, messages, n=1, temperature=1.0, max_new_tokens=2048,
                 prefill=None, seed=None):
        from tenacity import retry, stop_after_attempt, wait_exponential

        msgs = list(messages)
        if prefill is not None:
            # OpenRouter forwards a trailing assistant message as a prefill to
            # providers that support it.
            msgs = msgs + [{"role": "assistant", "content": prefill}]

        extra_body = {}
        if self.spec.disable_thinking:
            # OpenRouter unified knob; ignored by models that lack reasoning.
            extra_body["reasoning"] = {"enabled": False}

        @retry(stop=stop_after_attempt(5),
               wait=wait_exponential(multiplier=2, min=2, max=60))
        def _call():
            return self.client.chat.completions.create(
                model=self.spec.model_id,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=n,
                seed=seed,
                extra_body=extra_body or None,
            )

        resp = _call()
        return [c.message.content or "" for c in resp.choices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def load_backend(spec: ModelSpec, **kwargs) -> ModelBackend:
    if spec.backend == "hf":
        return HFBackend(spec, **kwargs)
    if spec.backend == "api":
        return APIBackend(spec, **kwargs)
    raise ValueError(f"unknown backend {spec.backend!r}")
