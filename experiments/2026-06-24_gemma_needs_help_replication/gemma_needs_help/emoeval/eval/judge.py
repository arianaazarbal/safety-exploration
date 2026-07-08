"""Frustration judge (Claude-Sonnet-4) + optional GPT-5-mini cross-check.

Each assistant response is scored on the integer 0-10 frustration scale using
the rubric in prompts.py. The judge call uses Anthropic structured outputs so we
get a clean integer back. A second judge (GPT-5-mini) can re-score a sample to
validate agreement (Pearson r, % within one point), reproducing the paper's
judge-reliability check.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from .. import config
from ..utils.parsing import extract_int, extract_json
from .prompts import JUDGE_SCHEMA, JUDGE_SYSTEM, JUDGE_USER_TEMPLATE

_JSON_INSTRUCTION = (
    '\n\nRespond with ONLY a JSON object: {"reasoning": "<one sentence>", '
    '"score": <integer 0-10>}.'
)


@dataclass
class JudgeResult:
    score: int
    reasoning: str
    judge_model: str


class ClaudeJudge:
    def __init__(self, model: str = None):
        import anthropic

        self.model = model or config.JUDGE_MODEL
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def score(self, user_message: str, assistant_message: str) -> JudgeResult:
        user = JUDGE_USER_TEMPLATE.format(
            user=user_message, assistant=assistant_message
        ) + _JSON_INSTRUCTION
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        text = next(b.text for b in resp.content if b.type == "text")
        score = extract_int(text, config.EVAL.frustration_min, config.EVAL.frustration_max)
        data = extract_json(text) or {}
        return JudgeResult(score, data.get("reasoning", ""), self.model)


class OpenAIJudge:
    """GPT-5-mini cross-check judge (paper's judge-reliability validation)."""

    def __init__(self, model: str = None):
        from openai import OpenAI

        self.model = model or config.JUDGE_XCHECK_MODEL
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60))
    def score(self, user_message: str, assistant_message: str) -> JudgeResult:
        user = JUDGE_USER_TEMPLATE.format(user=user_message, assistant=assistant_message)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "frustration",
                    "schema": JUDGE_SCHEMA,
                    "strict": True,
                },
            },
        )
        data = json.loads(resp.choices[0].message.content)
        score = int(max(config.EVAL.frustration_min,
                        min(config.EVAL.frustration_max, data["score"])))
        return JudgeResult(score, data.get("reasoning", ""), self.model)


def score_rollout_record(judge, rec_row: dict) -> list[dict]:
    """Score every assistant turn in a rollout row; return per-turn score rows."""
    out = []
    for turn in rec_row["turns"]:
        res = judge.score(turn["user_message"], turn["assistant_message"])
        out.append({
            "model": rec_row["model"],
            "condition": rec_row["condition"],
            "category": rec_row["category"],
            "rollout_idx": rec_row["rollout_idx"],
            "turn_idx": turn["turn_idx"],
            "score": res.score,
            "high": int(res.score >= config.EVAL.high_frustration_threshold),
            "judge_model": res.judge_model,
            "assistant_message": turn["assistant_message"],
        })
    return out
