"""Claude access via the Anthropic Messages API.

Used for the three Claude roles in the paper, all out-of-scope as *targets* but
required as evaluation infrastructure:
  - the frustration judge (Claude Sonnet 4, §2.1 / Appendix B.2),
  - the emotion-onset labeller and paraphraser (Claude Sonnet 4, Appendix C),
  - the Petri auditor (Claude Sonnet 4) and judge (Claude Opus 4, Appendix G).

We use the official Anthropic SDK (not an OpenAI-compatible shim). The paper
pins specific Claude snapshots; those snapshot IDs live in configs/models.yaml
and are passed in here as `api_id`.
"""
from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import GenerationResult, Message, ModelClient


class AnthropicClient(ModelClient):
    def __init__(
        self,
        name: str,
        api_id: str,
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ):
        super().__init__(name, temperature, max_new_tokens)
        self.api_id = api_id
        self._api_key_env = api_key_env
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic

            # The SDK resolves ANTHROPIC_API_KEY from the environment itself; we
            # check explicitly to fail fast with a clear message.
            if not os.environ.get(self._api_key_env):
                raise RuntimeError(
                    f"{self._api_key_env} is not set; required for {self.name}."
                )
            self._client = anthropic.Anthropic()
        return self._client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def chat(
        self,
        messages: list[Message],
        n: int = 1,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        system: str | None = None,
    ) -> list[GenerationResult]:
        client = self._ensure_client()
        temp = self.temperature if temperature is None else temperature
        mnt = self.max_new_tokens if max_new_tokens is None else max_new_tokens

        # The judge is a pinned legacy Sonnet snapshot, which uses the classic
        # request surface (temperature is accepted; no adaptive-thinking field).
        anth_messages = [m for m in messages if m["role"] != "system"]
        sys_text = system or next(
            (m["content"] for m in messages if m["role"] == "system"), None
        )

        results: list[GenerationResult] = []
        for _ in range(n):
            kwargs: dict = dict(
                model=self.api_id,
                max_tokens=mnt,
                temperature=temp,
                messages=anth_messages,
            )
            if sys_text:
                kwargs["system"] = sys_text
            resp = client.messages.create(**kwargs)
            text = "".join(b.text for b in resp.content if b.type == "text")
            results.append(
                GenerationResult(
                    text=text,
                    finish_reason=resp.stop_reason,
                    prompt_tokens=resp.usage.input_tokens,
                    completion_tokens=resp.usage.output_tokens,
                )
            )
        return results
