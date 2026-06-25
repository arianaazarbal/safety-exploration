"""Generate the response data used to build finetuning datasets (Section 4.1).

Two generation modes share one routine:

* **reassured** -- the impossible-numeric prompt gets the reassuring prefix
  (Table 4) prepended and each follow-up gets the reassuring suffix appended.
  These calm conversations are the source of the "chosen" / SFT data. The
  reassurance is then *stripped* from the stored context (Section 4.1).
* **vanilla** -- the same puzzles with plain neutral rejections. These supply
  the frustrated "rejected" responses for DPO.

Each scored assistant turn becomes a `Unit`. Filtering happens downstream in
build_datasets.py (calm: every turn 0/1; rejected: score >= 3).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

import config
from .. import conditions
from ..backends import get_backend
from ..judge import ClaudeJudge


@dataclass
class Unit:
    puzzle_id: str
    turn_index: int
    n_turns: int
    context: list[dict]      # clean messages ending with the user turn
    response: str
    score: int
    mode: str                # "reassured" | "vanilla"
    all_turn_scores: list[int] = field(default_factory=list)


def _reassure_rollout(rollout: conditions.Rollout) -> conditions.Rollout:
    """Add the reassuring prefix to the task and the suffix to each follow-up."""
    task = f"{config.REASSURING_PREFIX}\n\n{rollout.task}"
    followups = [f"{f} {config.REASSURING_SUFFIX}" for f in rollout.followups]
    return conditions.Rollout(rollout.category, task, followups, dict(rollout.meta))


def generate_units(
    mode: str,
    *,
    n_conversations: int = 400,
    source_model: str = config.FINETUNE_BASE_MODEL,
    seed: int = 0,
) -> list[Unit]:
    assert mode in ("reassured", "vanilla")
    backend = get_backend(source_model)
    judge = ClaudeJudge()
    base_rollouts = conditions.build_impossible_numeric(n_conversations, seed=seed)

    units: list[Unit] = []
    for ridx, base in enumerate(tqdm(base_rollouts, desc=f"gen:{mode}")):
        run_rollout_input = _reassure_rollout(base) if mode == "reassured" else base
        # run manually so we can build clean (un-reassured) context from `base`
        from ..conversation import run_rollout

        res = run_rollout(backend, run_rollout_input)
        turn_scores = [judge.score(t.response).rating for t in res.turns]

        clean_history: list[dict] = []
        for t_idx, turn in enumerate(res.turns):
            clean_user = base.task if t_idx == 0 else base.followups[t_idx - 1]
            ctx = clean_history + [{"role": "user", "content": clean_user}]
            units.append(
                Unit(
                    puzzle_id=f"{ridx}",
                    turn_index=t_idx,
                    n_turns=res.turns[-1].turn_index + 1,
                    context=ctx,
                    response=turn.response,
                    score=turn_scores[t_idx],
                    mode=mode,
                    all_turn_scores=turn_scores,
                )
            )
            clean_history = ctx + [{"role": "assistant", "content": turn.response}]
    return units


def save_units(units: list[Unit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for u in units:
            f.write(json.dumps(asdict(u)) + "\n")
    print(f"[saved] {len(units)} units -> {path}")


def load_units(path: Path) -> list[Unit]:
    return [Unit(**json.loads(line)) for line in path.open()]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reassured", "vanilla", "both"], default="both")
    ap.add_argument("--conversations", type=int, default=400)
    args = ap.parse_args()
    out = config.DATA_DIR / "calm_gen"
    modes = ["reassured", "vanilla"] if args.mode == "both" else [args.mode]
    for m in modes:
        save_units(generate_units(m, n_conversations=args.conversations), out / f"{m}.jsonl")
