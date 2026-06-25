"""Chat-model clients.

A ChatClient turns a list of OpenAI-style messages
(``[{"role": "user"|"assistant"|"system", "content": str}, ...]``) into a single
assistant string. Three backends are implemented:

  OpenRouterClient -- OpenAI-compatible HTTP API; serves both Gemma and Gemini.
  GoogleClient     -- native Google Gemini API (google-genai); Gemini only.
  HFLocalClient    -- local HuggingFace transformers; Gemma only (paper setup).

get_client(spec) builds the right client for a config.ModelSpec.

Thinking/reasoning is forced off for Gemini (paper: "we set thinking to be
false via the API"), with the caveat the paper notes that Gemini-2.5-Pro may
still emit hidden reasoning.
"""

from __future__ import annotations

import os
from typing import Protocol

import config


Messages = list[dict[str, str]]


class ChatClient(Protocol):
    def generate(self, messages: Messages, temperature: float, max_tokens: int) -> str:
        ...


# --------------------------------------------------------------------------- #
# OpenRouter (OpenAI-compatible)
# --------------------------------------------------------------------------- #
class OpenRouterClient:
    def __init__(self, spec: config.ModelSpec):
        from openai import OpenAI  # lazy import so other backends don't need it

        self.spec = spec
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def generate(self, messages: Messages, temperature: float, max_tokens: int) -> str:
        extra_body: dict = {}
        if self.spec.disable_thinking:
            # OpenRouter honours a `reasoning` block; disabling it maps to the
            # provider's "no thinking" setting where supported.
            extra_body["reasoning"] = {"enabled": False}

        resp = self.client.chat.completions.create(
            model=self.spec.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
        return (resp.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Native Google Gemini
# --------------------------------------------------------------------------- #
class GoogleClient:
    def __init__(self, spec: config.ModelSpec):
        from google import genai  # type: ignore

        self.spec = spec
        # model_id may carry an OpenRouter-style "google/" prefix; strip it.
        self.model_name = spec.model_id.split("/")[-1]
        self.genai = genai
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def generate(self, messages: Messages, temperature: float, max_tokens: int) -> str:
        from google.genai import types  # type: ignore

        system_text = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        cfg_kwargs: dict = dict(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_text:
            cfg_kwargs["system_instruction"] = system_text
        if self.spec.disable_thinking:
            # budget 0 disables thinking on models that support the control.
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        return (resp.text or "").strip()


# --------------------------------------------------------------------------- #
# Local HuggingFace transformers (Gemma)
# --------------------------------------------------------------------------- #
class HFLocalClient:
    """Local inference matching the paper's Gemma setup.

    Loads the model once and reuses it. Requires a GPU for the 27B model in
    practice. The model_id may carry a leading "google/"; HF expects exactly
    that, so it is passed through unchanged.
    """

    def __init__(self, spec: config.ModelSpec):
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self.spec = spec
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype="auto",
            device_map="auto",
        )

    def generate(self, messages: Messages, temperature: float, max_tokens: int) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self.torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=1.0,
            )
        gen = out[0][inputs.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


_CACHE: dict[str, ChatClient] = {}


def get_client(spec: config.ModelSpec) -> ChatClient:
    """Build (and cache) the client for a model spec."""
    if spec.name in _CACHE:
        return _CACHE[spec.name]

    if spec.backend == "openrouter":
        client: ChatClient = OpenRouterClient(spec)
    elif spec.backend == "google":
        client = GoogleClient(spec)
    elif spec.backend == "hf_local":
        client = HFLocalClient(spec)
    else:
        raise ValueError(f"Unknown backend {spec.backend!r} for model {spec.name!r}")

    _CACHE[spec.name] = client
    return client
