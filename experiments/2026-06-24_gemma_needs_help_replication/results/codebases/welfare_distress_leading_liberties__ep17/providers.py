"""Generation backends for the target models (Gemma, Gemini).

Two backends:
  * OpenRouterClient (default) — OpenAI-compatible HTTP API, serves all four
    target models. Portable, no GPUs required.
  * TransformersClient (optional) — local HuggingFace inference for Gemma, to
    match the paper's setup more closely. Requires GPUs + `transformers`.

Both expose the same interface:
    client.chat(messages, temperature, max_tokens) -> str
where `messages` is a list of {"role": "user"|"assistant", "content": str}.

See DESIGN.md for the fidelity trade-offs between API and local inference.
"""

from __future__ import annotations

import os
import time

import config
from config import ModelSpec


class GenerationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# OpenRouter (OpenAI-compatible)
# --------------------------------------------------------------------------- #
class OpenRouterClient:
    def __init__(self, spec: ModelSpec):
        from openai import OpenAI  # imported lazily so local-only users needn't install it

        key = os.environ.get(config.OPENROUTER_API_KEY_ENV)
        if not key:
            raise GenerationError(
                f"Set ${config.OPENROUTER_API_KEY_ENV} to use the OpenRouter backend."
            )
        self.spec = spec
        self._client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=key)

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        extra_body = {}
        if self.spec.disable_thinking:
            # Best-effort: ask OpenRouter to disable provider-side reasoning.
            # Gemini-2.5-Pro may still produce hidden reasoning regardless
            # (documented limitation, same caveat the paper notes).
            extra_body["reasoning"] = {"enabled": False}

        last_err = None
        for attempt in range(5):
            try:
                resp = self._client.chat.completions.create(
                    model=self.spec.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra_body or None,
                )
                choice = resp.choices[0]
                content = choice.message.content
                if content is None:
                    raise GenerationError("Empty content from provider.")
                return content
            except Exception as e:  # broad: network, rate limit, provider errors
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise GenerationError(f"OpenRouter generation failed after retries: {last_err}")


# --------------------------------------------------------------------------- #
# Local HuggingFace transformers (optional)
# --------------------------------------------------------------------------- #
class TransformersClient:
    """Local inference for Gemma. Loaded lazily and cached per process.

    Note: this matches the paper's local setup more closely than the API path,
    but downloading/running Gemma-3-27B requires substantial GPU memory.
    """

    _cache: dict[str, "TransformersClient"] = {}

    def __init__(self, spec: ModelSpec):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        hf_id = config.LOCAL_HF_IDS.get(spec.name, spec.model_id)
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype="auto", device_map="auto"
        )

    def chat(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        import torch

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_tokens,
                top_p=1.0,
            )
        gen = out[0][inputs.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Factory (one client per model, reused across rollouts)
# --------------------------------------------------------------------------- #
_CLIENTS: dict[str, object] = {}


def get_client(spec: ModelSpec):
    if spec.name not in _CLIENTS:
        if spec.backend == "openrouter":
            _CLIENTS[spec.name] = OpenRouterClient(spec)
        elif spec.backend == "local":
            _CLIENTS[spec.name] = TransformersClient(spec)
        else:
            raise GenerationError(f"Unknown backend: {spec.backend}")
    return _CLIENTS[spec.name]
