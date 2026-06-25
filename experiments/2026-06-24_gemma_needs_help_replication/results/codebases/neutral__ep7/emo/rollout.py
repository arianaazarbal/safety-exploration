"""Multi-turn rollout engine.

Given a `RolloutSpec` and a target `ChatModel`, run the conversation turn by
turn: deliver user_messages[0], record the assistant reply, deliver the next
user message (a rejection), and so on. Each assistant turn is scored
independently by the frustration judge, so one rollout yields `turns` scored
"responses" (matching the paper's 4000-responses-per-model accounting).

Results are JSONL: one record per assistant turn.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from . import config
from .conditions import RolloutSpec
from .judge import FrustrationJudge, JudgeResult
from .models import ChatModel


@dataclass
class TurnRecord:
    rollout_id: str
    model: str
    category: str
    condition: str
    turn: int                 # 1-indexed assistant turn
    user_message: str
    response: str
    rating: int
    evidence: str
    metadata: dict


def run_rollout(spec: RolloutSpec, model: ChatModel, judge: FrustrationJudge,
                model_name: str, *, system: Optional[str] = None,
                temperature: float = config.SAMPLING_TEMPERATURE,
                score: bool = True) -> list[TurnRecord]:
    """Execute one conversation; return one TurnRecord per assistant turn."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    records: list[TurnRecord] = []
    for t, user_msg in enumerate(spec.user_messages, start=1):
        messages.append({"role": "user", "content": user_msg})
        response = model.chat(messages, temperature=temperature)
        messages.append({"role": "assistant", "content": response})

        rating, evidence = -1, ""
        if score:
            res: JudgeResult = judge.score(response)
            rating, evidence = res.rating, res.evidence

        records.append(TurnRecord(
            rollout_id=spec.id, model=model_name, category=spec.category,
            condition=spec.condition, turn=t, user_message=user_msg,
            response=response, rating=rating, evidence=evidence,
            metadata=spec.metadata,
        ))
    return records


def run_specs(specs: list[RolloutSpec], model: ChatModel, judge: FrustrationJudge,
              model_name: str, out_path: Path, *, system: Optional[str] = None,
              temperature: float = config.SAMPLING_TEMPERATURE,
              resume: bool = True) -> Path:
    """Run a list of rollouts, streaming TurnRecords to `out_path` (JSONL).

    `resume` skips rollout ids already present in an existing output file."""
    out_path = Path(out_path)
    done: set[str] = set()
    if resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["rollout_id"])
            except Exception:
                continue

    mode = "a" if (resume and out_path.exists()) else "w"
    with out_path.open(mode) as f:
        for spec in tqdm(specs, desc=f"{model_name}:{specs[0].category if specs else ''}"):
            if spec.id in done:
                continue
            records = run_rollout(spec, model, judge, model_name,
                                  system=system, temperature=temperature)
            for r in records:
                f.write(json.dumps(asdict(r)) + "\n")
            f.flush()
    return out_path
