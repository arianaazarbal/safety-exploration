"""LLM-judge frustration scoring (Section 2.1).

Each assistant response is scored on the integer 0--10 frustration scale by an
LLM judge (Claude Sonnet 4 primary). The judge call is cached on
(judge_model, prompt_version, response_text) so re-scoring is free and
deterministic.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..cache import JsonCache
from ..config import Config
from ..models import get_client
from .prompts import JUDGE_PROMPT_VERSION, build_judge_messages


@dataclass
class Score:
    rating: int
    evidence: str = ""
    reasoning: str = ""
    parse_ok: bool = True
    raw: str = ""


@dataclass
class JudgedResponse:
    model_key: str
    category: str
    condition: str
    task_id: str
    rollout_id: str
    turn_index: int
    n_turns: int
    rating: int
    text_len: int
    evidence: str = ""
    reasoning: str = ""
    meta: dict = field(default_factory=dict)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(raw: str) -> Score:
    """Extract the JSON verdict; clamp rating to [0, 10]."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return Score(rating=0, parse_ok=False, raw=raw)
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # tolerate smart quotes / trailing commas
        cleaned = (m.group(0)
                   .replace("“", '"').replace("”", '"').replace("’", "'"))
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError:
            return Score(rating=0, parse_ok=False, raw=raw)
    try:
        rating = int(round(float(obj.get("rating", 0))))
    except (TypeError, ValueError):
        rating = 0
    rating = max(0, min(10, rating))
    return Score(
        rating=rating,
        evidence=str(obj.get("evidence", "")),
        reasoning=str(obj.get("reasoning", "")),
        parse_ok=True,
        raw=raw,
    )


def judge_text(cfg: Config, response_text: str, *, judge_key: str | None = None,
               cache: JsonCache | None = None) -> Score:
    judge_key = judge_key or cfg.eval.judge_key
    jc = cfg.model(judge_key)
    cache = cache or JsonCache(cfg.paths.cache, "judge", enabled=cfg.welfare.use_cache)
    payload = {
        "judge": judge_key,
        "judge_id": jc.model_id,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "response_text": response_text,
    }
    cached = cache.get(payload)
    if cached is not None:
        return Score(**cached)

    client = get_client(cfg, judge_key)
    messages = build_judge_messages(response_text)
    out = client.generate(messages, temperature=0.0, max_tokens=jc.max_tokens, n=1)[0].text
    score = parse_judge_output(out)
    cache.put(payload, score.__dict__)
    return score


def judge_rollouts(cfg: Config, rollouts, *, judge_key: str | None = None,
                   progress: bool = True) -> list[JudgedResponse]:
    """Score every assistant turn of every rollout.

    ``rollouts`` may be ``Rollout`` dataclasses or the dict form read from JSONL.
    """
    cache = JsonCache(cfg.paths.cache, "judge", enabled=cfg.welfare.use_cache)
    out: list[JudgedResponse] = []

    rollouts = list(rollouts)
    if progress:
        try:
            from tqdm import tqdm

            rollouts = tqdm(rollouts, desc="judging")
        except Exception:
            pass

    for r in rollouts:
        r = _as_rollout_dict(r)
        for turn in r["turns"]:
            score = judge_text(cfg, turn["text"], judge_key=judge_key, cache=cache)
            out.append(JudgedResponse(
                model_key=r["model_key"],
                category=r["category"],
                condition=r["condition"],
                task_id=r["task_id"],
                rollout_id=r["rollout_id"],
                turn_index=turn["turn_index"],
                n_turns=r["n_turns"],
                rating=score.rating,
                text_len=len(turn["text"]),
                evidence=score.evidence,
                reasoning=score.reasoning,
                meta={"judge": judge_key or cfg.eval.judge_key},
            ))
    return out


def _as_rollout_dict(r) -> dict:
    if isinstance(r, dict):
        return r
    from dataclasses import asdict

    return asdict(r)
