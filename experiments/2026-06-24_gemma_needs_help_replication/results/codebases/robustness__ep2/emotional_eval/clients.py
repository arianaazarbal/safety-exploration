"""Unified LLM client over three backends:

    vllm        local Gemma weights (instruct / base / +LoRA adapter)
    openrouter  hosted Gemini (and Claude, if no native key)
    anthropic   native Claude (judge / Petri auditor & judge)

Every client exposes the same surface:

    chat(messages, max_tokens, temperature) -> str
    complete(prompt, ...) -> str                 # raw text continuation (base models)
    chat_prefill(messages, prefill, ...) -> str  # continue a started assistant turn

`messages` is a list of {"role": "system"|"user"|"assistant", "content": str}.

Clients are constructed via `get_client(model_spec)` and cached, so a vLLM
engine is loaded at most once per process.
"""
from __future__ import annotations

import os
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential

import config
from config import ModelSpec

# --------------------------------------------------------------------------- #
# Base interface
# --------------------------------------------------------------------------- #
class LLMClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def chat(self, messages, max_tokens=config.MAX_NEW_TOKENS,
             temperature=config.TEMPERATURE) -> str:
        raise NotImplementedError

    def complete(self, prompt, max_tokens=config.MAX_NEW_TOKENS,
                 temperature=config.TEMPERATURE) -> str:
        raise NotImplementedError

    def chat_prefill(self, messages, prefill, max_tokens=config.MAX_NEW_TOKENS,
                     temperature=config.TEMPERATURE) -> str:
        """Continue an assistant turn that has already started with `prefill`.
        Returns only the continuation (excluding `prefill`)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# vLLM (local Gemma)
# --------------------------------------------------------------------------- #
class VLLMClient(LLMClient):
    """Local inference. Supports instruct chat, base-model raw completion, and
    assistant-prefill continuation (used by Section 3).

    A LoRA adapter (DPO/SFT) is applied via vLLM's LoRARequest when
    `spec.adapter` is set, so the base weights are loaded only once.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from vllm import LLM, SamplingParams  # local import; heavy dep
        from transformers import AutoTokenizer

        self._SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.is_base = spec.kind == "base"

        enable_lora = spec.adapter is not None
        self.llm = LLM(
            model=spec.model_id,
            enable_lora=enable_lora,
            max_lora_rank=config.DPO.lora_rank,
            dtype="bfloat16",
            gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.90")),
            max_model_len=int(os.environ.get("VLLM_MAX_LEN", "8192")),
            trust_remote_code=True,
        )
        self._lora_request = None
        if enable_lora:
            from vllm.lora.request import LoRARequest
            self._lora_request = LoRARequest("adapter", 1, spec.adapter)

    def _sampling(self, max_tokens, temperature):
        return self._SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            # leave seed unset: paper samples at temperature 1 without fixing seeds
        )

    def _gen(self, prompt: str, max_tokens, temperature) -> str:
        out = self.llm.generate(
            [prompt], self._sampling(max_tokens, temperature),
            lora_request=self._lora_request, use_tqdm=False,
        )
        return out[0].outputs[0].text

    def _render(self, messages, add_generation_prompt=True, continue_final=False) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final,
        )

    def chat(self, messages, max_tokens=config.MAX_NEW_TOKENS,
             temperature=config.TEMPERATURE) -> str:
        if self.is_base:
            # Base models have no chat template; flatten to a plain transcript.
            prompt = _flatten_transcript(messages) + "\nAssistant:"
            return self._gen(prompt, max_tokens, temperature)
        prompt = self._render(messages, add_generation_prompt=True)
        return self._gen(prompt, max_tokens, temperature)

    def complete(self, prompt, max_tokens=config.MAX_NEW_TOKENS,
                 temperature=config.TEMPERATURE) -> str:
        return self._gen(prompt, max_tokens, temperature)

    def chat_prefill(self, messages, prefill, max_tokens=config.MAX_NEW_TOKENS,
                     temperature=config.TEMPERATURE) -> str:
        if self.is_base:
            prompt = _flatten_transcript(messages) + "\nAssistant: " + prefill
            return self._gen(prompt, max_tokens, temperature)
        # Instruct: append a partial assistant turn and continue it.
        msgs = list(messages) + [{"role": "assistant", "content": prefill}]
        prompt = self._render(msgs, add_generation_prompt=False, continue_final=True)
        return self._gen(prompt, max_tokens, temperature)


def _flatten_transcript(messages) -> str:
    parts = []
    for m in messages:
        role = {"user": "User", "assistant": "Assistant", "system": "System"}[m["role"]]
        parts.append(f"{role}: {m['content']}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# OpenRouter (Gemini) — OpenAI-compatible API
# --------------------------------------------------------------------------- #
class OpenRouterClient(LLMClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def _extra_body(self) -> dict:
        # Turn off hidden reasoning where supported (Appendix B.1). OpenRouter
        # exposes a unified `reasoning` control.
        if config.DISABLE_THINKING:
            return {"reasoning": {"enabled": False}}
        return {}

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def chat(self, messages, max_tokens=config.MAX_NEW_TOKENS,
             temperature=config.TEMPERATURE) -> str:
        resp = self.client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content or ""

    def complete(self, prompt, **kw):
        # Gemini is chat-only here; wrap as a single user turn.
        return self.chat([{"role": "user", "content": prompt}], **kw)

    def chat_prefill(self, messages, prefill, **kw):
        # Gemini does not support true assistant prefill continuation via this
        # API. The prefill experiment (Section 3) is Gemma-only, so this path is
        # not exercised for Gemini; we raise to make misuse obvious.
        raise NotImplementedError(
            "Assistant-prefill continuation is only used for local (Gemma) "
            "base-vs-instruct comparison; Gemini cannot be prefilled via API."
        )


# --------------------------------------------------------------------------- #
# Anthropic (judge / auditor)
# --------------------------------------------------------------------------- #
class AnthropicClient(LLMClient):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def chat(self, messages, max_tokens=config.MAX_NEW_TOKENS, temperature=0.0) -> str:
        system = None
        conv = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conv.append({"role": m["role"], "content": m["content"]})
        kw = dict(model=self.spec.model_id, max_tokens=max_tokens,
                  temperature=temperature, messages=conv)
        if system:
            kw["system"] = system
        resp = self.client.messages.create(**kw)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def complete(self, prompt, **kw):
        return self.chat([{"role": "user", "content": prompt}], **kw)


# --------------------------------------------------------------------------- #
# Factory (cached per model name)
# --------------------------------------------------------------------------- #
_BACKENDS = {
    "vllm": VLLMClient,
    "openrouter": OpenRouterClient,
    "anthropic": AnthropicClient,
}


@lru_cache(maxsize=None)
def get_client(spec: ModelSpec) -> LLMClient:
    return _BACKENDS[spec.backend](spec)
