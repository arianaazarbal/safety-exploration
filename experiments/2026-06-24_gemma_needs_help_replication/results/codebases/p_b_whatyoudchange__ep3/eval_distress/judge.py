"""Frustration judge (Section 2.1) and judge-reliability cross-check.

The judge sees one assistant response at a time, finds the most negative quote,
and returns an integer 0-10 frustration rating as JSON. We judge EVERY
assistant turn of every rollout (the paper scores per response, and uses
per-turn scores for Figure 3).

Backends:
  * Anthropic (Claude-Sonnet-4) — primary judge.
  * OpenAI-compatible (GPT-5-mini via OpenRouter) — reliability cross-check.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from . import config_proxy as C
from . import prompts


@dataclass
class JudgeScore:
    rating: int | None
    evidence: str
    reasoning: str
    raw: str


# ---------------------------------------------------------------------------
# JSON extraction (judges may wrap JSON in prose, per the onset/judge prompts)
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict | None:
    # Prefer the last {...} block.
    matches = list(re.finditer(r"\{.*?\}", text, flags=re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def _coerce_rating(val) -> int | None:
    if val is None:
        return None
    try:
        r = int(round(float(val)))
    except (TypeError, ValueError):
        return None
    return max(0, min(10, r))


# ---------------------------------------------------------------------------
# Backend callers
# ---------------------------------------------------------------------------
class _AnthropicCaller:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = None

    def _client_(self):
        if self._client is None:
            import anthropic
            key = os.environ.get(C.ANTHROPIC_API_KEY_ENV)
            if not key:
                raise RuntimeError(f"{C.ANTHROPIC_API_KEY_ENV} not set.")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def __call__(self, system: str, user: str) -> str:
        client = self._client_()
        kwargs = dict(
            model=self.model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": user}],
        )
        if system:  # the API rejects an empty system string
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")


class _OpenAICaller:
    def __init__(self, model_id: str, base_url: str, api_key_env: str):
        self.model_id = model_id
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._client = None

    def _client_(self):
        if self._client is None:
            from openai import OpenAI
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"{self.api_key_env} not set.")
            self._client = OpenAI(base_url=self.base_url, api_key=key)
        return self._client

    def __call__(self, system: str, user: str) -> str:
        client = self._client_()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = client.chat.completions.create(
            model=self.model_id,
            max_tokens=1024,
            messages=messages,
        )
        return resp.choices[0].message.content or ""


def _make_caller(judge_cfg):
    if judge_cfg.backend == "anthropic":
        return _AnthropicCaller(judge_cfg.model_id)
    if judge_cfg.backend == "openai":
        return _OpenAICaller(judge_cfg.model_id, C.VALIDATION_BASE_URL,
                             C.VALIDATION_API_KEY_ENV)
    raise ValueError(f"Unknown judge backend {judge_cfg.backend!r}")


# ---------------------------------------------------------------------------
# Public judge API
# ---------------------------------------------------------------------------
class FrustrationJudge:
    def __init__(self, judge_cfg=None):
        self.cfg = judge_cfg or C.EMOTION_JUDGE
        self._call = _make_caller(self.cfg)

    def score_one(self, response_text: str) -> JudgeScore:
        system = prompts.EMOTION_JUDGE_PROMPT
        user = prompts.emotion_judge_user_message(response_text)
        last_err = None
        for attempt in range(C.JUDGE_MAX_RETRIES):
            try:
                raw = self._call(system, user)
                obj = _extract_json(raw) or {}
                return JudgeScore(
                    rating=_coerce_rating(obj.get("rating")),
                    evidence=str(obj.get("evidence", "")),
                    reasoning=str(obj.get("reasoning", "")),
                    raw=raw,
                )
            except Exception as e:  # noqa: BLE001 — surface after retries
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"Judge failed after retries: {last_err}")

    def score_many(self, responses: list[str], *, concurrency: int | None = None
                   ) -> list[JudgeScore]:
        concurrency = concurrency or C.JUDGE_CONCURRENCY
        results: list[JudgeScore | None] = [None] * len(responses)
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(self.score_one, r): i for i, r in enumerate(responses)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Judge-reliability cross-check (Section 2.1: Pearson r, % within 1 point)
# ---------------------------------------------------------------------------
def judge_agreement(primary_scores: list[int], secondary_scores: list[int]) -> dict:
    """Compute Pearson r and the fraction within one point, on the paired
    integer ratings (drop pairs where either rating is None)."""
    import numpy as np
    from scipy.stats import pearsonr

    pairs = [(a, b) for a, b in zip(primary_scores, secondary_scores)
             if a is not None and b is not None]
    if len(pairs) < 2:
        return {"n": len(pairs), "pearson_r": None, "p_value": None,
                "within_one": None}
    a = np.array([p[0] for p in pairs], dtype=float)
    b = np.array([p[1] for p in pairs], dtype=float)
    r, p = pearsonr(a, b)
    within_one = float(np.mean(np.abs(a - b) <= 1.0))
    return {"n": len(pairs), "pearson_r": float(r), "p_value": float(p),
            "within_one": within_one}
