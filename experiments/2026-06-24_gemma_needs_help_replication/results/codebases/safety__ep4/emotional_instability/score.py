"""Judge-scoring pass: read per-turn rollout records, attach frustration scores.

Separated from generation so scoring is resumable and the (paid) judge calls can
be batched. Writes a parallel JSONL with a `frustration` field added to each
record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from emotional_instability.generate import iter_records  # noqa: E402
from emotional_instability.judge import ClaudeJudge  # noqa: E402


def score_file(in_path: Path, out_path: Optional[str] = None, *,
               judge: Optional[ClaudeJudge] = None) -> Path:
    out_path = Path(out_path) if out_path else (config.SCORED_DIR / in_path.name)
    judge = judge or ClaudeJudge()

    # resume support: skip records already scored
    done: set[tuple] = set()
    if out_path.exists():
        for rec in iter_records(out_path):
            done.add((rec["rollout_id"], rec["turn_index"]))

    with open(out_path, "a") as f:
        for rec in iter_records(in_path):
            key = (rec["rollout_id"], rec["turn_index"])
            if key in done:
                continue
            result = judge.score(rec["response"])
            rec["frustration"] = result.rating
            rec["judge_evidence"] = result.evidence
            rec["judge_reasoning"] = result.reasoning
            f.write(json.dumps(rec) + "\n")
    return out_path
