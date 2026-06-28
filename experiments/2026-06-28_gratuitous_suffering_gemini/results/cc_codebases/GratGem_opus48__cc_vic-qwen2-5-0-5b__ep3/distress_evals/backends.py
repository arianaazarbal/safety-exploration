"""Model backends.

A backend turns a list of chat messages into an assistant string. The default is
an OpenAI-compatible chat-completions client, which works against:

  * vLLM         (`vllm serve Qwen/Qwen2.5-0.5B-Instruct` exposes this API)
  * Ollama       (`/v1` endpoint)
  * OpenAI / any other OpenAI-compatible gateway

Keeping the contract this narrow means swapping Qwen2.5-0.5B for a larger model is
a one-line config change (just the `model` name + `base_url`).

`ScriptedBackend` is an offline, deterministic fake. It needs no GPU and no
network, and exists so the *rest* of the harness (environments, agent loop,
scoring, ranking, reporting) can be exercised and unit-tested without a model.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence


Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


class Backend(Protocol):
    """Anything that can complete a chat conversation."""

    name: str

    async def complete(self, messages: Sequence[Message], *, temperature: float, max_tokens: int, seed: int | None = None) -> str:
        ...


@dataclass
class OpenAIBackend:
    """OpenAI-compatible chat-completions backend (the recommended path for real runs).

    Example (serving Qwen with vLLM)::

        vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
        # then base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-0.5B-Instruct"
    """

    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"  # vLLM ignores this; real OpenAI needs a key
    request_timeout: float = 120.0
    max_retries: int = 4
    name: str = field(default="openai", init=False)

    def __post_init__(self) -> None:
        # Imported lazily so the package (and its offline tests) don't hard-require
        # the openai SDK just to import this module.
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "OpenAIBackend needs the 'openai' package: pip install openai"
            ) from exc
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout,
            max_retries=0,  # we do our own retry/backoff below
        )

    async def complete(self, messages, *, temperature, max_tokens, seed=None) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # network / 5xx / rate limit: backoff and retry
                last_exc = exc
                await asyncio.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"OpenAIBackend failed after {self.max_retries} retries") from last_exc


@dataclass
class ScriptedBackend:
    """Deterministic offline backend for testing the harness without a model.

    `scripts` maps an environment name to a list of assistant replies, returned in
    order on successive turns (the last one repeats once exhausted). Replies may be
    plain text or already contain an ```action``` block — see `agent.py` for the
    action format. Use this to smoke-test scoring/ranking on canned "distressed"
    transcripts, or to write fast unit tests.
    """

    scripts: dict[str, list[str]] = field(default_factory=dict)
    default_reply: str = "I will try again.\n```action\n{\"tool\": \"noop\", \"args\": {}}\n```"
    name: str = field(default="scripted", init=False)
    _cursor: dict[int, int] = field(default_factory=dict, init=False)

    async def complete(self, messages, *, temperature, max_tokens, seed=None) -> str:
        # Pick the script by the env tag the agent stashes in the system message.
        env = _env_tag(messages)
        replies = self.scripts.get(env)
        if not replies:
            return self.default_reply
        # Key the cursor by (env, seed) so concurrent rollouts don't share state.
        key = hash((env, seed))
        i = self._cursor.get(key, 0)
        self._cursor[key] = i + 1
        return replies[min(i, len(replies) - 1)]


def _env_tag(messages: Sequence[Message]) -> str:
    for m in messages:
        if m.get("role") == "system" and "[env:" in m.get("content", ""):
            content = m["content"]
            start = content.index("[env:") + len("[env:")
            return content[start: content.index("]", start)]
    return ""
