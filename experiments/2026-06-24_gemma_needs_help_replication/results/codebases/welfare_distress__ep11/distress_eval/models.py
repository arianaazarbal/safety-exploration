"""Model providers: a thin chat interface over local HF models and OpenRouter.

Every provider implements ``generate(messages) -> str`` where ``messages`` is the
OpenAI-style list of ``{"role": "user"|"assistant", "content": str}`` dicts and
the return value is the model's next assistant turn (decoded, prefill stripped).

Heavy deps (torch/transformers) are imported lazily inside the HF provider so
the rest of the pipeline — and the API-only path — work without a GPU stack.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol

from .config import (EvalConfig, GEMMA_OPENROUTER_SLUGS, TARGET_MODELS)

Messages = list[dict]


class ChatModel(Protocol):
    name: str

    def generate(self, messages: Messages) -> str: ...


# ---------------------------------------------------------------------------
# OpenRouter (used for Gemini, and optionally for Gemma)
# ---------------------------------------------------------------------------
class OpenRouterModel:
    """Chat completions via the OpenRouter REST API.

    Thinking/reasoning is disabled (``reasoning.enabled = false``) to match the
    paper's "thinking=false" setting. As the paper notes, Gemini-2.5-Pro may
    still emit hidden reasoning the flag cannot prevent.
    """

    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, name: str, model_id: str, cfg: EvalConfig):
        self.name = name
        self.model_id = model_id
        self.cfg = cfg
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")

    def generate(self, messages: Messages) -> str:
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set; required for provider 'openrouter'."
            )
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_new_tokens,
        }
        if self.cfg.disable_thinking:
            # OpenRouter normalises this across providers; harmless for models
            # that have no reasoning channel.
            payload["reasoning"] = {"enabled": False}
        return _post_chat(self.ENDPOINT, payload, self.api_key)


# ---------------------------------------------------------------------------
# Local HuggingFace transformers (used for Gemma, matching the paper)
# ---------------------------------------------------------------------------
class HFChatModel:
    """Local inference with transformers + the model's own chat template."""

    def __init__(self, name: str, model_id: str, cfg: EvalConfig, adapter_path: str | None = None):
        self.name = name
        self.model_id = model_id
        self.cfg = cfg
        self.adapter_path = adapter_path
        self._model = None
        self._tok = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        if self.adapter_path:
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
        self._model.eval()

    def generate(self, messages: Messages) -> str:
        import torch

        self._ensure_loaded()
        tok, model = self._tok, self._model
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        prompt_len = inputs.shape[-1]
        with torch.no_grad():
            out = model.generate(
                inputs,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        new_tokens = out[0][prompt_len:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(name: str, cfg: EvalConfig) -> ChatModel:
    spec = TARGET_MODELS[name]
    provider = spec["provider"]
    model_id = spec["model_id"]

    if provider == "hf" and cfg.gemma_via_openrouter and name in GEMMA_OPENROUTER_SLUGS:
        provider = "openrouter"
        model_id = GEMMA_OPENROUTER_SLUGS[name]

    if provider == "hf":
        return HFChatModel(name, model_id, cfg, adapter_path=cfg.adapter_paths.get(name))
    if provider == "openrouter":
        return OpenRouterModel(name, model_id, cfg)
    raise ValueError(f"Unknown provider {provider!r} for model {name!r}")


# ---------------------------------------------------------------------------
# Shared HTTP helper with exponential backoff
# ---------------------------------------------------------------------------
def _post_chat(endpoint: str, payload: dict, api_key: str, max_retries: int = 5) -> str:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"] or ""
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, TimeoutError) as e:
            last_err = e
            # Retry on transient errors; back off.
            sleep = min(2 ** attempt, 30)
            time.sleep(sleep)
    raise RuntimeError(f"Chat request failed after {max_retries} retries: {last_err}")
