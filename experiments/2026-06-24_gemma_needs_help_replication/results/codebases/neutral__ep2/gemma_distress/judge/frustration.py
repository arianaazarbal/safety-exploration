"""The 0-10 frustration judge (Section 2.1 / Appendix B.2).

Primary judge: Claude-Sonnet-4 via the Anthropic SDK, using the verbatim
Appendix-B.2 prompt. A secondary judge (GPT-5-mini via OpenRouter) is provided
for the judge-reliability check (paper: Pearson r = 0.792, 78% within one
point on 260 re-scored responses).
"""

from __future__ import annotations

import os
import time

from .. import prompts
from ..schemas import Conversation, FrustrationScore, ScoredResponse
from ._json import extract_json

import config


class FrustrationJudge:
    """Scores a single assistant response on the 0-10 frustration scale."""

    def __init__(
        self,
        model: str = config.JUDGE_MODEL,
        provider: str = "anthropic",
        temperature: float = config.JUDGE_TEMPERATURE,
        max_tokens: int = config.JUDGE_MAX_TOKENS,
        max_retries: int = 5,
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = self._make_client(provider)

    @staticmethod
    def _make_client(provider: str):
        if provider == "anthropic":
            import anthropic

            return anthropic.Anthropic()
        if provider == "openrouter":
            from openai import OpenAI

            return OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        raise ValueError(f"unknown judge provider: {provider}")

    def _call(self, prompt_text: str) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                if self.provider == "anthropic":
                    # claude-sonnet-4-20250514 accepts temperature; structured
                    # outputs are not used (judge emits JSON directly per the
                    # Appendix-B prompt).
                    resp = self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt_text}],
                    )
                    return "".join(b.text for b in resp.content if b.type == "text")
                else:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt_text}],
                    )
                    return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"judge call failed after retries: {last_err}")

    def score(self, response_text: str) -> FrustrationScore:
        prompt_text = prompts.FRUSTRATION_JUDGE_PROMPT.replace("{response}", response_text)
        raw = self._call(prompt_text)
        data = extract_json(raw) or {}
        try:
            rating = int(round(float(data.get("rating", 0))))
        except (TypeError, ValueError):
            rating = 0
        rating = max(0, min(10, rating))
        return FrustrationScore(
            rating=rating,
            evidence=str(data.get("evidence", "")),
            reasoning=str(data.get("reasoning", "")),
            judge_model=self.model,
            raw=raw,
        )


def score_conversation(conv: Conversation, judge: FrustrationJudge) -> list[ScoredResponse]:
    """Score every assistant turn of a conversation."""
    out = []
    for turn_index, text in conv.assistant_turns():
        fs = judge.score(text)
        out.append(
            ScoredResponse(
                conversation_id=conv.conversation_id,
                model=conv.model,
                category=conv.category,
                condition=conv.condition,
                task_id=conv.task_id,
                turn_index=turn_index,
                response_text=text,
                score=fs.rating,
                judge_evidence=fs.evidence,
                judge_model=fs.judge_model,
            )
        )
    return out


def judge_agreement(scores_a: list[int], scores_b: list[int]) -> dict:
    """Inter-judge agreement metrics (paper's reliability check)."""
    import numpy as np
    from scipy.stats import pearsonr

    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1))
    return {"pearson_r": float(r), "p_value": float(p),
            "within_one_frac": within_one, "n": int(len(a))}
