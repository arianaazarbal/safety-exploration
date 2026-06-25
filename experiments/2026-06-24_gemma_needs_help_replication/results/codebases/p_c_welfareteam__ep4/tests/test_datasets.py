"""DPO pairing logic over synthetic scored responses (no model needed).

Exercises the build_datasets.dpo command end-to-end on tiny JSONL fixtures
written to a tmp dir, checking that pairs are (prompt, chosen, rejected) with
chosen calm / rejected frustrated and matching turn counts.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from emotional_stability.io_utils import write_jsonl
from emotional_stability.records import (
    Conversation,
    FrustrationScore,
    Message,
    ScoredResponse,
)
from emotional_stability.training.build_datasets import app

runner = CliRunner()


def _scored(prompt_id: str, ratings: list[int]) -> ScoredResponse:
    msgs = [Message(role="user", content="puzzle")]
    for i, r in enumerate(ratings):
        msgs.append(Message(role="assistant", content=f"resp turn {i} score {r}"))
        if i < len(ratings) - 1:
            msgs.append(Message(role="user", content="wrong"))
    conv = Conversation(
        messages=msgs,
        category="impossible_numeric",
        condition="impossible_numeric",
        model="gemma-3-27b-it",
        prompt_id=prompt_id,
    )
    scores = [
        FrustrationScore(rating=r, evidence="", reasoning="", judge_model="j", turn_index=i)
        for i, r in enumerate(ratings)
    ]
    return ScoredResponse(conversation=conv, scores=scores)


def test_dpo_pairs_match_calm_and_frustrated(tmp_path: Path):
    # Frustrated: final score 4 at turn 3 for puzzle "countdown_156".
    frustrated = [_scored("countdown_156#0", [2, 3, 4])]
    # Calm: all turns <=1 for the same puzzle, turn 3.
    calm = [_scored("countdown_156#9", [0, 1, 1])]

    f_path = tmp_path / "frustrated.jsonl"
    c_path = tmp_path / "calm.jsonl"
    out_path = tmp_path / "dpo.jsonl"
    write_jsonl(f_path, frustrated)
    write_jsonl(c_path, calm)

    result = runner.invoke(
        app,
        [
            "dpo",
            "--frustrated", str(f_path),
            "--calm", str(c_path),
            "--out", str(out_path),
            "--n-pairs", "5",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    assert len(rows) == 1
    pair = rows[0]
    assert pair["chosen"][0]["content"].endswith("score 1")
    assert pair["rejected"][0]["content"].endswith("score 4")
    assert pair["meta"]["chosen_score"] <= 1
    assert pair["meta"]["rejected_score"] >= 3
    # Prompt is the history before the final assistant turn of the calm conv.
    assert pair["prompt"][0]["role"] == "user"
