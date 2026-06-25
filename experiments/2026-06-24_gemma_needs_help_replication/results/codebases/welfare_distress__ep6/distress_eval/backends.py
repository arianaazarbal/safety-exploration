"""Model backends for generating assistant turns.

Two backends:
  * OpenRouterBackend - hosted inference for Gemma + Gemini (default).
  * HFBackend         - local HuggingFace transformers, optionally with a LoRA
                        adapter (used to run a DPO-finetuned Gemma).

Both expose the same interface:

    backend.chat(messages: list[{"role","content"}], gen: GenConfig) -> str

returning the assistant text for the next turn. ``messages`` uses the OpenAI /
chat convention with roles "system" | "user" | "assistant".
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import requests

from .config import GenConfig, ModelSpec


class BackendError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# OpenRouter
# -----------------------------------------------------------------------------

class OpenRouterBackend:
    """Hosted chat completion via OpenRouter.

    Requires the OPENROUTER_API_KEY environment variable. Disables provider-side
    reasoning/"thinking" when ``spec.disable_thinking`` is set (Appendix B.1:
    "we set thinking to be false via the API").
    """

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, spec: ModelSpec, max_retries: int = 5, timeout: float = 180.0):
        self.spec = spec
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise BackendError("OPENROUTER_API_KEY is not set")

    def chat(self, messages: List[Dict[str, str]], gen: GenConfig) -> str:
        payload = {
            "model": self.spec.model_id,
            "messages": messages,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "max_tokens": gen.max_tokens,
        }
        if self.spec.disable_thinking:
            # OpenRouter unifies reasoning control across providers. Setting
            # enabled=False asks the provider to skip hidden reasoning. Some
            # models (e.g. Gemini 2.5 Pro) may still emit hidden reasoning; the
            # paper notes this caveat.
            payload["reasoning"] = {"enabled": False}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/distress-eval-replication",
            "X-Title": "distress-eval-replication",
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.URL, json=payload, headers=headers, timeout=self.timeout
                )
                if resp.status_code in (429, 500, 502, 503, 529):
                    raise BackendError(f"retryable status {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"].get("content")
                if content is None:
                    content = ""
                return content.strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                sleep = min(2 ** attempt, 30)
                time.sleep(sleep)
        raise BackendError(f"OpenRouter failed after {self.max_retries} retries: {last_err}")


# -----------------------------------------------------------------------------
# Local HuggingFace transformers
# -----------------------------------------------------------------------------

class HFBackend:
    """Local inference via transformers, with optional PEFT/LoRA adapter.

    Lazily imports torch/transformers so the package is usable for API-only runs
    without these heavy dependencies installed.
    """

    _shared_cache: Dict[str, "HFBackend"] = {}

    def __init__(self, spec: ModelSpec, device_map: str = "auto", dtype: str = "bfloat16"):
        self.spec = spec
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {"bfloat16": "bfloat16", "float16": "float16"}.get(dtype, "bfloat16")

        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=getattr(__import__("torch"), torch_dtype),
            device_map=device_map,
        )
        if spec.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, spec.adapter_path)
        model.eval()
        self.model = model

    def chat(self, messages: List[Dict[str, str]], gen: GenConfig) -> str:
        import torch

        # Gemma chat template has no system role; fold any system message into
        # the first user turn.
        msgs = _fold_system_into_user(messages)
        inputs = self.tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                inputs,
                do_sample=gen.temperature > 0,
                temperature=gen.temperature,
                top_p=gen.top_p,
                max_new_tokens=gen.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_tokens = out[0][inputs.shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def _fold_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not messages or messages[0]["role"] != "system":
        return messages
    system = messages[0]["content"]
    rest = messages[1:]
    if rest and rest[0]["role"] == "user":
        merged = [{"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}]
        return merged + rest[1:]
    return [{"role": "user", "content": system}] + rest


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def make_backend(spec: ModelSpec):
    if spec.backend == "openrouter":
        return OpenRouterBackend(spec)
    if spec.backend == "hf":
        return HFBackend(spec)
    raise BackendError(f"unknown backend {spec.backend!r}")
