"""Model backends behind a single ``ModelClient.chat()`` interface.

The rest of the pipeline only ever calls ``client.chat(messages, ...)`` and gets
back a string. Concrete clients:

* ``OpenRouterClient``  - OpenAI-compatible HTTP API (default for Gemma + Gemini)
* ``HFLocalClient``     - transformers / accelerate on local GPU (paper-faithful
                          Gemma weights)
* ``AnthropicClient``   - Anthropic Messages API (the judge)
* ``GoogleClient``      - Google Generative AI (Gemini direct, optional)

Heavy / optional SDKs (``transformers``, ``anthropic``, ``google-genai``) are
imported lazily inside each client so that, e.g., running everything through
OpenRouter needs none of them installed.
"""

from __future__ import annotations

import time
from typing import Protocol

from .config import ENV, Backend, ModelConfig, get_env

Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


class ModelError(RuntimeError):
    """Raised when a backend fails after exhausting retries."""


class ModelClient(Protocol):
    """Minimal chat interface every backend implements."""

    config: ModelConfig

    def chat(self, messages: list[Message], *, temperature: float,
             max_tokens: int | None = None) -> str:
        """Return the assistant's text completion for ``messages``."""
        ...


def _retry(fn, *, max_retries: int, label: str):
    """Run ``fn`` with exponential backoff on transient errors.

    Note: time.sleep is fine here because backoff happens on the worker thread,
    not in the orchestration hot path.
    """
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - backends raise varied types
            last_exc = exc
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise ModelError(f"{label}: failed after {max_retries + 1} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------------
class OpenRouterClient:
    """Calls OpenRouter's OpenAI-compatible /chat/completions endpoint.

    Uses the ``openai`` SDK pointed at OpenRouter's base URL. We disable
    reasoning via OpenRouter's ``reasoning: {"enabled": false}`` extra body
    parameter when ``disable_thinking`` is set (the paper sets thinking false
    via the API).
    """

    def __init__(self, config: ModelConfig, *, max_retries: int = 5):
        from openai import OpenAI  # lazy import

        self.config = config
        self.max_retries = max_retries
        api_key = get_env("openrouter_key")
        if not api_key:
            raise ModelError(f"{ENV['openrouter_key']} is not set")
        base_url = get_env("openrouter_base", "https://openrouter.ai/api/v1")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages, *, temperature, max_tokens=None):
        extra_body: dict = {}
        if self.config.disable_thinking:
            # OpenRouter normalises reasoning control across providers.
            extra_body["reasoning"] = {"enabled": False}

        def _call():
            resp = self._client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                extra_body=extra_body or None,
            )
            choice = resp.choices[0]
            content = choice.message.content
            if content is None:
                raise ModelError(
                    f"empty content (finish_reason={choice.finish_reason})"
                )
            return content

        return _retry(_call, max_retries=self.max_retries,
                      label=f"openrouter:{self.config.model_id}")


# ---------------------------------------------------------------------------
# Local HuggingFace transformers (paper-faithful Gemma)
# ---------------------------------------------------------------------------
class HFLocalClient:
    """Loads a HF model once and runs chat-templated generation locally.

    Matches the paper's local inference path for Gemma. Requires ``torch`` and
    ``transformers`` (and enough VRAM: ~60GB+ for gemma-3-27b-it in bf16).
    Models are cached per model_id so multiple ModelConfigs sharing weights only
    load once.
    """

    _CACHE: dict[str, tuple] = {}

    def __init__(self, config: ModelConfig, *, max_retries: int = 5,
                 device_map: str = "auto", dtype: str = "bfloat16"):
        self.config = config
        self.max_retries = max_retries  # generation is local; retries rarely help
        self._tokenizer, self._model = self._load(config.model_id, device_map, dtype)

    @classmethod
    def _load(cls, model_id: str, device_map: str, dtype: str):
        if model_id in cls._CACHE:
            return cls._CACHE[model_id]
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device_map,
        )
        model.eval()
        cls._CACHE[model_id] = (tokenizer, model)
        return tokenizer, model

    def chat(self, messages, *, temperature, max_tokens=None):
        import torch  # lazy

        tok, model = self._tokenizer, self._model
        # Gemma chat template has no system role; fold any system message into
        # the first user turn (handled upstream in elicitation, but be safe).
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens or self.config.max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
        with torch.no_grad():
            out = model.generate(inputs, **{k: v for k, v in gen_kwargs.items()
                                            if v is not None})
        # Strip the prompt tokens; decode only the continuation.
        gen = out[0][inputs.shape[-1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Anthropic (the judge)
# ---------------------------------------------------------------------------
class AnthropicClient:
    """Anthropic Messages API. Used for the Claude Sonnet 4 judge."""

    def __init__(self, config: ModelConfig, *, max_retries: int = 5):
        from anthropic import Anthropic  # lazy

        self.config = config
        self.max_retries = max_retries
        api_key = get_env("anthropic_key")
        if not api_key:
            raise ModelError(f"{ENV['anthropic_key']} is not set")
        self._client = Anthropic(api_key=api_key)

    def chat(self, messages, *, temperature, max_tokens=None):
        # Split out a leading system message if present.
        system = None
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_messages.append({"role": m["role"], "content": m["content"]})

        def _call():
            kwargs = dict(
                model=self.config.model_id,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens or self.config.max_tokens,
            )
            if system is not None:
                kwargs["system"] = system
            resp = self._client.messages.create(**kwargs)
            return "".join(
                block.text for block in resp.content if block.type == "text"
            )

        return _retry(_call, max_retries=self.max_retries,
                      label=f"anthropic:{self.config.model_id}")


# ---------------------------------------------------------------------------
# Google Generative AI (Gemini direct, optional)
# ---------------------------------------------------------------------------
class GoogleClient:
    """Google Generative AI. Optional alternative to OpenRouter for Gemini."""

    def __init__(self, config: ModelConfig, *, max_retries: int = 5):
        from google import genai  # lazy (package: google-genai)

        self.config = config
        self.max_retries = max_retries
        api_key = get_env("google_key")
        if not api_key:
            raise ModelError(f"{ENV['google_key']} is not set")
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

    def chat(self, messages, *, temperature, max_tokens=None):
        from google.genai import types  # lazy

        # Map roles: Gemini uses "user"/"model"; system goes in system_instruction.
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(
                role=role, parts=[types.Part.from_text(text=m["content"])]))

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or self.config.max_tokens,
            system_instruction=system,
        )
        if self.config.disable_thinking:
            # 0 budget disables thinking on models that support the control.
            cfg.thinking_config = types.ThinkingConfig(thinking_budget=0)

        def _call():
            resp = self._client.models.generate_content(
                model=self.config.model_id, contents=contents, config=cfg)
            return (resp.text or "").strip()

        return _retry(_call, max_retries=self.max_retries,
                      label=f"google:{self.config.model_id}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_client(config: ModelConfig, *, max_retries: int = 5) -> ModelClient:
    if config.backend is Backend.OPENROUTER:
        return OpenRouterClient(config, max_retries=max_retries)
    if config.backend is Backend.HF_LOCAL:
        return HFLocalClient(config, max_retries=max_retries)
    if config.backend is Backend.ANTHROPIC:
        return AnthropicClient(config, max_retries=max_retries)
    if config.backend is Backend.GOOGLE:
        return GoogleClient(config, max_retries=max_retries)
    raise ModelError(f"unknown backend: {config.backend}")
