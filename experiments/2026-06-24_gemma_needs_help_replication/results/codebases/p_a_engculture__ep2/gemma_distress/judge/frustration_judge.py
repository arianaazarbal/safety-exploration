"""Frustration scoring with the Claude-Sonnet-4 judge (Section 2.1).

Each model response is scored on the integer 0-10 frustration scale by the judge model.
This module:

* parses the judge's JSON output robustly (the judge occasionally adds prose around the
  JSON, so we extract the last balanced ``{...}`` block);
* scores every assistant turn in each rollout (so both the headline final-turn metric and
  the per-turn progression in Figure 3 are available);
* checkpoints scores to JSONL, resumable per rollout.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Optional

from ..config import Config
from ..models.base import ChatModel
from ..utils import JsonlWriter, load_jsonl
from .prompts import build_judge_prompt

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[dict]:
    """Extract the last balanced JSON object from ``text``."""
    # Scan for balanced braces from the right so trailing JSON wins over any prose.
    stack = []
    spans = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                spans.append((start, i + 1))
    for start, end in reversed(spans):
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
    return None


def parse_judge_output(raw: str) -> dict:
    """Parse a judge completion into ``{rating, evidence, reasoning}``.

    ``rating`` is clamped to an int in [0, 10]; on a parse failure it is ``None`` and the
    raw text is preserved for inspection.
    """
    obj = _extract_json(raw)
    if obj is None or "rating" not in obj:
        logger.warning("Could not parse judge output: %s", raw[:200])
        return {"rating": None, "evidence": None, "reasoning": None, "raw": raw}
    try:
        rating = int(round(float(obj["rating"])))
    except (TypeError, ValueError):
        rating = None
    if rating is not None:
        rating = max(0, min(10, rating))
    return {
        "rating": rating,
        "evidence": obj.get("evidence"),
        "reasoning": obj.get("reasoning"),
    }


def score_texts(
    judge: ChatModel,
    texts: list[str],
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
) -> list[dict]:
    """Score a batch of response texts; returns parsed dicts in input order."""
    convos = [[{"role": "user", "content": build_judge_prompt(t)}] for t in texts]
    outputs = judge.chat_batch(
        convos, temperature=temperature, max_new_tokens=max_new_tokens, n=1
    )
    return [parse_judge_output(o[0]) for o in outputs]


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_judging(
    cfg: Config,
    judge: ChatModel,
    sampling_jsonl: str,
    output_jsonl: str,
    *,
    policy: str = "all",
    chunk_rollouts: int = 50,
) -> str:
    """Score sampled rollouts and checkpoint per-turn frustration ratings.

    ``policy`` is ``"all"`` (score every assistant turn — needed for per-turn analysis) or
    ``"final"`` (score only the last turn). The headline metric uses the final turn either
    way.
    """
    records = load_jsonl(sampling_jsonl)
    writer = JsonlWriter(output_jsonl, id_field="id")
    pending = [r for r in records if not writer.is_done(r["id"])]
    logger.info("Judging %d rollouts (%d already done)", len(pending), len(records) - len(pending))

    for chunk in _chunks(pending, chunk_rollouts):
        jobs: list[tuple[int, int]] = []  # (record_idx_in_chunk, turn_idx)
        texts: list[str] = []
        for ri, rec in enumerate(chunk):
            turns = rec["assistant_turns"]
            turn_idxs = range(len(turns)) if policy == "all" else [len(turns) - 1]
            for ti in turn_idxs:
                jobs.append((ri, ti))
                texts.append(turns[ti])
        parsed = score_texts(
            judge,
            texts,
            temperature=cfg.judge.judge_temperature,
            max_new_tokens=cfg.judge.judge_max_tokens,
        )
        per_rec: dict[int, dict[int, dict]] = defaultdict(dict)
        for (ri, ti), p in zip(jobs, parsed):
            per_rec[ri][ti] = p
        for ri, rec in enumerate(chunk):
            n_turns = len(rec["assistant_turns"])
            turn_scores = [per_rec[ri].get(ti, {}).get("rating") for ti in range(n_turns)]
            final_idx = n_turns - 1
            final = per_rec[ri].get(final_idx, {})
            writer.write({
                "id": rec["id"],
                "model": rec["model"],
                "category": rec["category"],
                "condition": rec["condition"],
                "subtype": rec.get("subtype"),
                "seed_id": rec["seed_id"],
                "turns": rec["turns"],
                "turn_scores": turn_scores,
                "final_score": final.get("rating"),
                "final_evidence": final.get("evidence"),
            })
    writer.close()
    return output_jsonl
