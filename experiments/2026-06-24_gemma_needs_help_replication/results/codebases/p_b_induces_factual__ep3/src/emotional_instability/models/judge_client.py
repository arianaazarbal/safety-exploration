"""Thin API clients for the auxiliary models: the Claude judge / auditor /
onset-labeller / paraphraser, and the GPT secondary judge.

These are deliberately separate from :class:`ChatModel`: they are utilities used
*by* the harness (scoring, paraphrasing) rather than evaluation targets, and they
have a simpler call surface (a single user message, optionally a system prompt).
The OpenAI client also doubles as an OpenRouter client when ``base_url`` is set,
which is how the paper accessed several API models.
"""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..logging_utils import get_logger

logger = get_logger(__name__)


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 1024, temperature: float = 0.0):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        return self._client

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def complete(self, prompt: str, system: str | None = None) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def converse(self, messages: list[dict], system: str | None = None) -> str:
        """Multi-turn variant used by the Petri auditor."""
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIClient:
    """Used for the GPT secondary judge (gpt-5-mini) and OpenRouter access."""

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key_env = api_key_env
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=os.environ.get(self.api_key_env),
                base_url=self.base_url,
            )
        return self._client

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def build_aux_client(spec) -> AnthropicClient | OpenAIClient:
    """Construct an auxiliary client from a config node (``judge``, ``paraphraser``…)."""
    backend = spec.backend
    if backend == "anthropic":
        return AnthropicClient(
            model=spec.model,
            max_tokens=spec.get("max_tokens", 1024),
            temperature=spec.get("temperature", 0.0),
        )
    if backend == "openai":
        return OpenAIClient(model=spec.model, base_url=spec.get("base_url"))
    raise ValueError(f"unknown auxiliary backend: {backend}")
