"""OpenAI (GPT) judge backend — secondary validation judge (paper: GPT-5-mini).

Used only to re-score the 260-response agreement subsample (§2.1). Built on the
official ``openai`` SDK. Works against the OpenAI API or any OpenAI-compatible
endpoint (set OPENAI_BASE_URL).
"""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import JudgeSpec
from .base import Judge


class OpenAIJudge(Judge):
    def __init__(self, spec: JudgeSpec):
        super().__init__(spec)

    @property
    def _client(self):
        from openai import OpenAI

        # Resolves OPENAI_API_KEY (and optional OPENAI_BASE_URL) from the env.
        return OpenAI()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=max_tokens or self.spec.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
