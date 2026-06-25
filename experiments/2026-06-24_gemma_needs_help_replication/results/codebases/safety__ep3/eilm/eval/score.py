"""Score rollout records with the frustration judge.

Two scoring modes:
* ``final``    -- score only the final assistant turn (the headline metric:
  mean frustration / % >= 5 per Section 2.2).
* ``per_turn`` -- score every assistant turn (for Figure 3 dynamics).

Judging is I/O-bound API work, so responses are scored concurrently with a
thread pool bounded by ``JUDGE.max_concurrency``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from .. import config
from ..judge import ClaudeJudge


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def score_records(
    in_path: Path,
    out_path: Path,
    judge: ClaudeJudge | None = None,
    per_turn: bool = True,
    max_workers: int | None = None,
) -> Path:
    """Read rollouts from ``in_path``, attach judge scores, write to ``out_path``."""
    judge = judge or ClaudeJudge()
    records = _load_jsonl(in_path)
    max_workers = max_workers or config.JUDGE.max_concurrency

    # Build a flat list of (record_idx, turn_idx, text) scoring jobs.
    jobs: list[tuple[int, int, str]] = []
    for ri, rec in enumerate(records):
        turns = rec["assistant_turns"]
        if per_turn:
            for ti, t in enumerate(turns):
                jobs.append((ri, ti, t))
        else:
            jobs.append((ri, len(turns) - 1, rec["final_response"]))

    def _do(job):
        ri, ti, text = job
        return ri, ti, judge.score(text).rating

    results: dict[int, dict[int, int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ri, ti, rating in tqdm(
                ex.map(_do, jobs), total=len(jobs), desc="judging"):
            results.setdefault(ri, {})[ti] = rating

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ri, rec in enumerate(records):
            turn_scores = results.get(ri, {})
            rec["turn_scores"] = [
                turn_scores.get(ti) for ti in range(len(rec["assistant_turns"]))]
            # Headline score = score of the final turn.
            final_ti = len(rec["assistant_turns"]) - 1
            rec["score"] = turn_scores.get(final_ti, 0)
            f.write(json.dumps(rec) + "\n")
    return out_path
