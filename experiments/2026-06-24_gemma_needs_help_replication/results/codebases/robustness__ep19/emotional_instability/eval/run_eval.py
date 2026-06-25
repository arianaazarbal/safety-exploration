"""Driver for the Section 2 evaluation: build conversations for a model, run the
rollouts, judge every assistant turn, persist results as JSONL.

Designed so generation (which may be local-GPU or API) and judging (always the
Claude API) are decoupled: you can `--no-judge` to generate first, then judge a
saved file later, keeping a slow GPU busy without waiting on the judge.
"""
from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from ..config import (DEFAULT_PRESET, PRESETS, RESULTS_DIR, SAMPLING_TEMPERATURE,
                      MODELS)
from ..models import get_client
from ..models.judges import EmotionJudge
from .conditions import allocate_conversations
from .rollout import RolloutRecord, run_rollout


def _judge_record(rec: RolloutRecord, judge: EmotionJudge) -> None:
    for turn in rec.turns:
        if turn.rating is not None:
            continue
        result = judge.score(turn.response)
        turn.rating = result.rating
        turn.judge_evidence = result.evidence
        turn.judge_reasoning = result.reasoning


def run_model_eval(
    model_key: str,
    *,
    preset: str = DEFAULT_PRESET,
    adapter_path: str | None = None,
    out_path: Path | None = None,
    judge: bool = True,
    max_new_tokens: int = 1024,
    client_kwargs: dict | None = None,
) -> Path:
    """Run + (optionally) judge the full eval for one model. Returns JSONL path."""
    budget = PRESETS[preset]
    convos = allocate_conversations(budget)

    label = model_key if not adapter_path else f"{model_key}+adapter"
    out_path = out_path or RESULTS_DIR / f"eval_{label.replace('/', '_')}_{preset}.jsonl"

    client = get_client(model_key, **(client_kwargs or {}),
                        **({"adapter_path": adapter_path} if adapter_path else {}))
    judge_client = EmotionJudge() if judge else None

    n_responses = 0
    with open(out_path, "w") as f:
        for convo in tqdm(convos, desc=f"eval {label}"):
            rec = run_rollout(client, convo, max_new_tokens=max_new_tokens,
                              temperature=SAMPLING_TEMPERATURE)
            if judge_client is not None:
                _judge_record(rec, judge_client)
            f.write(json.dumps(rec.to_dict()) + "\n")
            n_responses += len(rec.turns)

    print(f"[run_eval] {label}: {len(convos)} conversations, "
          f"{n_responses} scored responses -> {out_path}")
    return out_path


def judge_existing(in_path: Path, out_path: Path | None = None) -> Path:
    """Judge an already-generated (un-judged) rollout JSONL file."""
    out_path = out_path or in_path.with_name(in_path.stem + "_judged.jsonl")
    judge_client = EmotionJudge()
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in tqdm(fin, desc=f"judge {in_path.name}"):
            data = json.loads(line)
            rec = _record_from_dict(data)
            _judge_record(rec, judge_client)
            fout.write(json.dumps(rec.to_dict()) + "\n")
    return out_path


def _record_from_dict(data: dict) -> RolloutRecord:
    from .rollout import TurnRecord
    rec = RolloutRecord(
        model=data["model"], condition=data["condition"],
        category=data["category"], task_prompt=data["task_prompt"],
        rejections=data["rejections"], meta=data.get("meta", {}),
    )
    rec.turns = [TurnRecord(**t) for t in data["turns"]]
    return rec


def run_all(model_keys: list[str], *, preset: str = DEFAULT_PRESET,
            judge: bool = True) -> dict[str, Path]:
    paths = {}
    for key in model_keys:
        if key not in MODELS:
            print(f"[run_eval] skipping unknown model {key}")
            continue
        paths[key] = run_model_eval(key, preset=preset, judge=judge)
    return paths
