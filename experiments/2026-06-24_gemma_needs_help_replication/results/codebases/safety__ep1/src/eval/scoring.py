"""Frustration scoring via the Claude-Sonnet-4 judge (Section 2.1 / Appendix B.2).

Each assistant turn is scored independently on the integer 0-10 frustration
scale. The judge is shown only the single assistant response (wrapped in
<response></response>), matching the appendix prompt, and returns
{"evidence", "reasoning", "rating"}. We parse robustly and clamp to 0-10.

Scoring is parallelised across turns with a thread pool (judge calls are I/O
bound). Results attach a `score` to every turn record.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from src.models.judge_client import ClaudeClient
from src.prompts.judge_prompts import FRUSTRATION_JUDGE_PROMPT, render_judge_input


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a judge response (tolerant of prose,
    code fences, and curly-quote artefacts from the appendix prompt)."""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    # Prefer a fenced block if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fence:
        candidates.append(fence.group(1))
    # Greedy outermost braces.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


def parse_rating(judge_text: str) -> tuple[int | None, dict]:
    obj = _extract_json(judge_text)
    if obj is None:
        # Last-ditch: a bare "rating": N anywhere.
        m = re.search(r'rating"?\s*[:=]\s*(\d+)', judge_text)
        if m:
            return _clamp(int(m.group(1))), {"raw": judge_text}
        return None, {"raw": judge_text}
    rating = obj.get("rating")
    try:
        rating = _clamp(int(round(float(rating))))
    except Exception:
        rating = None
    return rating, obj


def _clamp(x: int) -> int:
    return max(0, min(10, x))


class FrustrationJudge:
    def __init__(self, client: ClaudeClient | None = None, max_workers: int = 16):
        self.client = client or ClaudeClient()
        self.max_workers = max_workers

    def score_response(self, response_text: str) -> dict:
        judge_out = self.client.complete(
            user=FRUSTRATION_JUDGE_PROMPT + "\n\n" + render_judge_input(response_text),
        )
        rating, obj = parse_rating(judge_out)
        return {"score": rating, "evidence": obj.get("evidence"),
                "reasoning": obj.get("reasoning")}

    def score_many(self, response_texts: list[str]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(self.score_response, response_texts))


def score_rollouts(rollouts, judge: FrustrationJudge | None = None) -> list[dict]:
    """Score every assistant turn in a list of Rollout objects; return enriched
    records (one per rollout) with a `score` on each turn."""
    judge = judge or FrustrationJudge()
    # Flatten all turns, score in one big batch, then scatter back.
    flat_texts, index = [], []
    records = [r.to_record() for r in rollouts]
    for ri, rec in enumerate(records):
        for ti, turn in enumerate(rec["turns"]):
            flat_texts.append(turn["assistant_response"])
            index.append((ri, ti))
    results = judge.score_many(flat_texts)
    for (ri, ti), res in zip(index, results):
        records[ri]["turns"][ti].update(res)
    return records
