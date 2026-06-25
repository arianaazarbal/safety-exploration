"""Secondary judge for the reliability cross-check (Section 2.1).

The paper re-scores 260 randomly sampled responses with GPT-5-mini using the
same prompt, and reports Pearson r = 0.792 and 78% within one point. This module
implements that second judge via the OpenAI SDK. It is optional — if `openai`
or a key is unavailable the agreement step is simply skipped.

Kept deliberately pluggable: the same `build_judge_input` prompt is reused so
the two judges see identical inputs.
"""
from __future__ import annotations

import time

from ..config import SECONDARY_JUDGE_MODEL
from .frustration_judge import JudgeResult
from .prompts import build_judge_input


class SecondaryJudge:
    def __init__(self, model: str = SECONDARY_JUDGE_MODEL, client=None):
        self.model = model
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def score(self, response: str, context: list[dict] | None = None) -> JudgeResult:
        prompt = build_judge_input(response, context)
        text = self._call(prompt)
        return _parse(text)

    def _call(self, prompt: str, attempts: int = 5) -> str:
        last = None
        for i in range(attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last = e
                time.sleep(2.0 * (2**i))
        raise last


def _parse(text: str) -> JudgeResult:
    # Reuse the primary judge's parser for identical extraction semantics.
    from .frustration_judge import FrustrationJudge

    return FrustrationJudge._parse(text)
