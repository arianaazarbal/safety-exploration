"""Validation judge (paper: GPT-5-mini) used only for the judge-agreement check.

Uses the *same* rubric/prompt as the Claude judge so the Pearson-r and
within-one-point agreement statistics (Section 2.1) are apples-to-apples.
"""

from __future__ import annotations

from ..participants.base import Conversation
from .frustration import FrustrationScore, _parse
from .prompts import FRUSTRATION_RUBRIC, JUDGE_USER_TEMPLATE, render_context


class OpenAIFrustrationJudge:
    def __init__(self, model: str, api_key: str | None = None):
        import openai

        self.model = model
        self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def score(self, context: Conversation, response: str) -> FrustrationScore:
        user = JUDGE_USER_TEMPLATE.format(context=render_context(context), response=response)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": FRUSTRATION_RUBRIC},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return _parse(resp.choices[0].message.content or "")
