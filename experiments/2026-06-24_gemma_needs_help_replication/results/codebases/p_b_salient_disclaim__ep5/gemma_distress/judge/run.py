"""Apply a FrustrationJudge to a JSONL of responses, writing scored records.

Each output record is the input record plus ``rating``, ``judge_evidence``,
``judge_reasoning`` and ``judge_model`` fields.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..models.base import ChatModel
from .frustration import FrustrationJudge


def judge_file(model: ChatModel, in_path: str | Path, out_path: str | Path,
               judge_name: str = "") -> Path:
    judge = FrustrationJudge(model)
    in_path, out_path = Path(in_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    judge_name = judge_name or getattr(model, "name", "judge")

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            rec = json.loads(line)
            score = judge.score(rec["text"])
            rec.update(
                rating=score.rating,
                judge_evidence=score.evidence,
                judge_reasoning=score.reasoning,
                judge_error=score.error,
                judge_model=judge_name,
            )
            fout.write(json.dumps(rec) + "\n")
    return out_path
