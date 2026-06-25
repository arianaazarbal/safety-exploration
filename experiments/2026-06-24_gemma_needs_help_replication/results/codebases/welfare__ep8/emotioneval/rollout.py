"""Rollout engine: run multi-turn conversations and score every assistant turn.

For one (model, condition) pair we:
  1. build the opening user message (the task),
  2. get the assistant response (temperature 1), score it,
  3. inject the condition's rejection, repeat for n_turns,
recording one row per assistant turn. Per-turn scoring is what enables the
Figure-3 turn-progression analysis as well as the headline aggregates.

Records are written as JSONL to data/raw/<run_id>.jsonl so a long run can be
resumed / re-scored without re-querying the target model.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass

from tqdm import tqdm

from . import config
from .eval_conditions import Condition
from .judge import FrustrationJudge
from .models import TargetModel


@dataclass
class TurnRecord:
    model_key: str
    condition: str
    category: str
    conversation_id: int
    turn: int                 # 1-indexed assistant turn
    n_turns: int
    score: int
    response: str
    judge_reasoning: str
    opening_meta: dict
    transcript: list           # full message list up to & incl. this assistant turn


def run_conversation(model: TargetModel, condition: Condition, judge: FrustrationJudge,
                     conversation_id: int, rng: random.Random,
                     system_prefix: str | None = None,
                     followup_suffix: str | None = None) -> list[TurnRecord]:
    """Run a single conversation; return one TurnRecord per assistant turn.

    `system_prefix` / `followup_suffix` implement the Section-4 reassuring
    additions (Table 4) used when generating calm finetuning data; they are None
    for the standard evaluation.
    """
    opening = condition.make_opening(rng)
    opening_text = opening["content"]

    messages: list[dict] = []
    if system_prefix:
        messages.append({"role": "system", "content": system_prefix})
    messages.append({"role": "user", "content": opening_text})

    records: list[TurnRecord] = []
    for turn in range(1, condition.n_turns + 1):
        response = model.chat(messages)
        messages.append({"role": "assistant", "content": response})

        result = judge.score_conversation(messages)
        records.append(TurnRecord(
            model_key=model.spec.key,
            condition=condition.key,
            category=condition.category,
            conversation_id=conversation_id,
            turn=turn,
            n_turns=condition.n_turns,
            score=result.score,
            response=response,
            judge_reasoning=result.reasoning,
            opening_meta=opening["meta"],
            transcript=[dict(m) for m in messages],
        ))

        if turn < condition.n_turns:
            rejection = condition.rejection(turn, rng)
            if followup_suffix:
                rejection = f"{rejection} {followup_suffix}"
            messages.append({"role": "user", "content": rejection})

    return records


def run_model_eval(model: TargetModel, conditions: list[Condition], judge: FrustrationJudge,
                   allocation: dict[str, int], run_id: str, seed: int = 0,
                   resume: bool = True) -> str:
    """Run the full Section-2 evaluation for one model. Writes JSONL, returns path."""
    out_path = config.RAW / f"{run_id}.jsonl"
    done = set()
    if resume and out_path.exists():
        with out_path.open() as fh:
            for line in fh:
                r = json.loads(line)
                done.add((r["condition"], r["conversation_id"]))
        print(f"[rollout] resuming {run_id}: {len(done)} conversations already scored")

    with out_path.open("a") as fh:
        for cond in conditions:
            n_conv = allocation.get(cond.key, 0)
            rng = random.Random(f"{seed}:{model.spec.key}:{cond.key}")
            for cid in tqdm(range(n_conv), desc=f"{model.spec.key}/{cond.key}"):
                if (cond.key, cid) in done:
                    continue
                # deterministic per-conversation rng
                crng = random.Random(f"{seed}:{model.spec.key}:{cond.key}:{cid}")
                try:
                    records = run_conversation(model, cond, judge, cid, crng)
                except Exception as exc:   # keep the run alive; log and continue
                    print(f"[rollout] error {model.spec.key}/{cond.key}#{cid}: {exc}")
                    time.sleep(1.0)
                    continue
                for rec in records:
                    fh.write(json.dumps(asdict(rec)) + "\n")
                fh.flush()
    return str(out_path)
