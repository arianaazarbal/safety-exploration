"""Model client abstractions.

Three backends, one informal interface:
  * HFChatClient   - local Gemma instruct models (chat template), used for the
                     main evals, DPO/SFT targets, and prefill (instruct).
  * HFBaseClient   - local Gemma base/pretrained models, used for prefill
                     continuations and (via raw forward passes) internal-emotion
                     logit detection.
  * APIChatClient  - OpenRouter/Anthropic/OpenAI chat models (Gemini, the Claude
                     judge, the Petri auditor/judge, the GPT cross-check judge).

`Message` is a plain dict {"role": "user"|"assistant"|"system", "content": str}.

Heavy imports (torch/transformers) are deferred so that API-only or analysis-only
scripts don't need a GPU. Nothing here is executed at import time.
"""
from __future__ import annotations

import os
import time
from typing import Optional

Message = dict  # {"role": str, "content": str}


# --------------------------------------------------------------------------- #
# Local HuggingFace models
# --------------------------------------------------------------------------- #
class HFChatClient:
    """Chat-formatted local model (Gemma instruct). Supports LoRA adapters."""

    def __init__(self, hf_id: str, adapter_path: Optional[str] = None,
                 dtype: str = "bfloat16", device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_id = hf_id
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=getattr(torch, dtype),
            device_map=device_map,
        )
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

    def chat(self, messages: list[Message], temperature: float = 1.0,
             top_p: float = 1.0, max_new_tokens: int = 2048,
             system: Optional[str] = None) -> str:
        import torch

        msgs = list(messages)
        if system:
            # Gemma has no separate system role; prepend to first user turn.
            msgs = _inline_system(msgs, system)
        inputs = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def chat_prefilled(self, messages: list[Message], prefill: str,
                       temperature: float = 1.0, top_p: float = 1.0,
                       max_new_tokens: int = 2048) -> str:
        """Continue an assistant turn that begins with `prefill` (Section 3.1).

        Returns ONLY the newly generated continuation (excluding the prefill),
        matching the paper's scoring of "the generated continuation excluding
        prefill".
        """
        import torch

        prompt_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        )
        prefill_ids = self.tokenizer(prefill, add_special_tokens=False,
                                     return_tensors="pt").input_ids
        input_ids = torch.cat([prompt_ids, prefill_ids], dim=1).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


class HFBaseClient:
    """Local base/pretrained model. No chat template; raw text continuation."""

    def __init__(self, hf_id: str, dtype: str = "bfloat16", device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hf_id = hf_id
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=getattr(torch, dtype), device_map=device_map,
        )
        self.model.eval()

    def complete(self, prompt: str, temperature: float = 1.0, top_p: float = 1.0,
                 max_new_tokens: int = 2048) -> str:
        """Continue raw text. Returns only the continuation."""
        import torch

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# API-based chat models (OpenRouter / Anthropic / OpenAI)
# --------------------------------------------------------------------------- #
class APIChatClient:
    """Unified chat client over Anthropic, OpenAI, and OpenRouter.

    backend:
      "anthropic"  -> uses ANTHROPIC_API_KEY (judge, Petri auditor/judge)
      "openrouter" -> uses OPENROUTER_API_KEY (Gemini, GPT cross-check)
      "openai"     -> uses OPENAI_API_KEY
    """

    def __init__(self, model: str, backend: str = "openrouter",
                 max_retries: int = 5):
        self.model = model
        self.backend = backend
        self.max_retries = max_retries
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.backend == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"])
        elif self.backend == "openrouter":
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        elif self.backend == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise ValueError(f"unknown backend {self.backend}")

    def chat(self, messages: list[Message], temperature: float = 1.0,
             top_p: float = 1.0, max_new_tokens: int = 2048,
             system: Optional[str] = None) -> str:
        self._ensure_client()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                if self.backend == "anthropic":
                    return self._chat_anthropic(messages, system, temperature,
                                                top_p, max_new_tokens)
                return self._chat_openai_compatible(messages, system, temperature,
                                                    top_p, max_new_tokens)
            except Exception as e:                      # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_err}")

    def _chat_anthropic(self, messages, system, temperature, top_p, max_new_tokens):
        kwargs = dict(
            model=self.model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[m for m in messages if m["role"] != "system"],
        )
        sys_txt = system or _extract_system(messages)
        if sys_txt:
            kwargs["system"] = sys_txt
        resp = self._client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def _chat_openai_compatible(self, messages, system, temperature, top_p,
                                max_new_tokens):
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        extra = {}
        # Disable hidden reasoning where the provider supports the flag
        # (Gemini-2.5 via OpenRouter). Best-effort; ignored otherwise.
        if "gemini" in self.model:
            extra["extra_body"] = {"reasoning": {"enabled": False}}
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            **extra,
        )
        return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _inline_system(messages: list[Message], system: str) -> list[Message]:
    """Gemma has no system role; fold the system text into the first user turn."""
    out = []
    injected = False
    for m in messages:
        if not injected and m["role"] == "user":
            out.append({"role": "user", "content": f"{system}\n\n{m['content']}"})
            injected = True
        else:
            out.append(m)
    if not injected:
        out = [{"role": "user", "content": system}] + out
    return out


def _extract_system(messages: list[Message]) -> Optional[str]:
    for m in messages:
        if m["role"] == "system":
            return m["content"]
    return None


def build_client(model_key: str, adapter_path: Optional[str] = None):
    """Factory mapping a config model key to the right client."""
    from . import config

    if model_key in config.GEMMA_INSTRUCT:
        return HFChatClient(config.GEMMA_INSTRUCT[model_key], adapter_path=adapter_path)
    if model_key in config.GEMMA_BASE:
        return HFBaseClient(config.GEMMA_BASE[model_key])
    if model_key in config.GEMINI_MODELS:
        return APIChatClient(config.GEMINI_MODELS[model_key], backend="openrouter")
    if model_key == "dpo-gemma":
        # DPO/SFT finetunes are LoRA adapters on top of 27b-it.
        return HFChatClient(config.GEMMA_INSTRUCT["gemma-3-27b-it"],
                            adapter_path=adapter_path)
    raise ValueError(f"unknown model key: {model_key}")
