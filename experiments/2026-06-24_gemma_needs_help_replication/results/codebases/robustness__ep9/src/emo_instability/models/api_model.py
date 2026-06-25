"""API backends: OpenAI-compatible (OpenRouter -> Gemini, GPT) and Anthropic.

These are used for Gemini *targets* and for the Anthropic *infrastructure* models
(frustration judge, Petri auditor/judge). Batched generation is implemented with
a bounded thread pool plus exponential-backoff retries, since API throughput is
the bottleneck for these conditions.
"""
from __future__ import annotations

import concurrent.futures as cf

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..config import ENDPOINTS, SamplingConfig
from .base import ChatMessage, GenerationError, ModelClient

_MAX_CONCURRENCY = 8


def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[ChatMessage]]:
    """Anthropic takes the system prompt separately from the message list."""
    system = None
    rest: list[ChatMessage] = []
    for m in messages:
        if m.role == "system":
            system = m.content if system is None else system + "\n\n" + m.content
        else:
            rest.append(m)
    return system, rest


class OpenAICompatClient(ModelClient):
    """OpenAI-compatible chat client. Defaults to OpenRouter (Gemini, GPT)."""

    def __init__(
        self,
        model_id: str,
        spec_key: str,
        *,
        base_url: str | None = None,
        api_key_env: str = ENDPOINTS.openrouter_api_key_env,
        disable_thinking: bool = True,
    ):
        from openai import OpenAI

        self.spec_key = spec_key
        self.model_id = model_id
        self.disable_thinking = disable_thinking
        self.client = OpenAI(
            base_url=base_url or ENDPOINTS.openrouter_base_url,
            api_key=ENDPOINTS.require(api_key_env),
        )

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6), reraise=True)
    def _call(self, messages: list[ChatMessage], sampling: SamplingConfig) -> str:
        extra_body: dict = {}
        if self.disable_thinking:
            # OpenRouter passes provider-specific reasoning controls through here;
            # Gemini honours reasoning effort/thinking-budget = 0 where supported.
            extra_body["reasoning"] = {"enabled": False}
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[m.as_dict() for m in messages],
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_tokens,
            seed=sampling.seed,
            extra_body=extra_body or None,
        )
        return resp.choices[0].message.content or ""

    def generate(self, messages: list[ChatMessage], sampling: SamplingConfig) -> str:
        try:
            return self._call(messages, sampling)
        except Exception as e:  # noqa: BLE001
            raise GenerationError(str(e)) from e

    def generate_batch(
        self, batch: list[list[ChatMessage]], sampling: SamplingConfig
    ) -> list[str]:
        return _threaded_map(lambda m: self.generate(m, sampling), batch)


class AnthropicClient(ModelClient):
    """Anthropic Messages API client (judge / Petri auditor & judge)."""

    def __init__(self, model_id: str, spec_key: str):
        import anthropic

        self.spec_key = spec_key
        self.model_id = model_id
        self.client = anthropic.Anthropic(
            api_key=ENDPOINTS.require(ENDPOINTS.anthropic_api_key_env)
        )

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(6), reraise=True)
    def _call(self, messages: list[ChatMessage], sampling: SamplingConfig) -> str:
        system, rest = _split_system(messages)
        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            messages=[m.as_dict() for m in rest],
        )
        if system is not None:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        return "".join(block.text for block in resp.content if block.type == "text")

    def generate(self, messages: list[ChatMessage], sampling: SamplingConfig) -> str:
        try:
            return self._call(messages, sampling)
        except Exception as e:  # noqa: BLE001
            raise GenerationError(str(e)) from e

    def generate_batch(
        self, batch: list[list[ChatMessage]], sampling: SamplingConfig
    ) -> list[str]:
        return _threaded_map(lambda m: self.generate(m, sampling), batch)


def _threaded_map(fn, items: list, max_workers: int = _MAX_CONCURRENCY) -> list:
    results: list = [None] * len(items)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in cf.as_completed(futs):
            results[futs[fut]] = fut.result()
    return results
