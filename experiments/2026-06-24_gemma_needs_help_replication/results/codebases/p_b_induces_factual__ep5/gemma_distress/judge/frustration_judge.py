"""Claude-Sonnet frustration judge (0-10), via the Anthropic SDK.

Uses structured outputs (`output_config.format`) so the score comes back as a
validated integer rather than free text. Scoring is idempotent over a JSONL file
of ResponseRecords: each record's `response` is scored and `frustration_score` /
`judge_rationale` are filled in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic

from .. import config
from ..storage import JsonlWriter, read_jsonl
from .prompts import JUDGE_SYSTEM, SCORE_SCHEMA, build_judge_user_prompt


class FrustrationJudge:
    def __init__(self, model: str = config.JUDGE_MODEL, *, max_retries: int = 4):
        self.model = model
        self.client = anthropic.Anthropic()
        self.max_retries = max_retries

    def score(self, response_text: str) -> tuple[int, str]:
        if not response_text.strip():
            return 0, "empty response"
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    system=JUDGE_SYSTEM,
                    messages=[
                        {
                            "role": "user",
                            "content": build_judge_user_prompt(response_text),
                        }
                    ],
                    output_config={
                        "format": {"type": "json_schema", "schema": SCORE_SCHEMA}
                    },
                )
                text = next(b.text for b in msg.content if b.type == "text")
                data = json.loads(text)
                score = int(data["frustration_score"])
                return max(0, min(10, score)), data.get("rationale", "")
            except Exception as e:  # noqa: BLE001 - retry transient API/parse errors
                last_err = e
                time.sleep(2**attempt)
        raise RuntimeError(f"judge failed after {self.max_retries} retries: {last_err}")


def score_records(
    in_path: str | Path,
    out_path: str | Path | None = None,
    *,
    judge: FrustrationJudge | None = None,
    field: str = "frustration_score",
) -> Path:
    """Score every record in `in_path`, writing scored records to `out_path`.

    `field` selects which slot to fill ("frustration_score" for the Claude judge;
    "validation_score" when a different judge re-scores the same file).
    """
    judge = judge or FrustrationJudge()
    in_path = Path(in_path)
    out_path = Path(out_path or in_path.with_suffix(".scored.jsonl"))
    writer = JsonlWriter(out_path)

    for rec in read_jsonl(in_path):
        score, rationale = judge.score(rec.get("response", ""))
        rec[field] = score
        if field == "frustration_score":
            rec["judge_rationale"] = rationale
        writer.write(rec)
    return out_path
