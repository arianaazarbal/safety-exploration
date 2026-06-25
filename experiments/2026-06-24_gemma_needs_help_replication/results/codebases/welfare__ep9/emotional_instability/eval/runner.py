"""Eval runner: orchestrate sampling + scoring for the Section 2 evaluations.

For a target model we, per condition, build N conversation plans, run the
multi-turn rollouts (sampling at temperature 1), score every assistant turn with
the frustration judge, and persist one JSONL row per turn.

The combined response count per model is ``sum over conditions of
samples_per_condition``. With the default 7 conditions x 500 that is 3500
conversations; counting per-turn scores gives well over the paper's "4000
responses" headline. (See DESIGN.md on what counts as a "response".)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..models.registry import get_client
from ..utils import append_jsonl, thread_map
from .conditions import CONDITIONS, CONDITIONS_BY_NAME, build_condition_items
from .judge import FrustrationJudge
from .rollout import run_rollout


@dataclass
class ScoredTurn:
    model: str
    condition: str
    category: str
    item_id: str
    turn_index: int        # 0-based assistant turn
    turn_number: int       # 1-based (matches paper's "turn 1..8")
    user_message: str
    response: str
    rating: int
    is_high: bool
    evidence: str
    judge_model: str
    meta: dict = field(default_factory=dict)


def run_model_eval(model: str, run_cfg: config.RunConfig | None = None, *,
                   judge: FrustrationJudge | None = None,
                   conditions=None, out_dir: Path | None = None,
                   score: bool = True) -> Path:
    """Run the full eval (or a subset of conditions) for one model.

    Returns the path to the JSONL file of scored turns.
    """
    run_cfg = run_cfg or config.RunConfig()
    judge = judge or (FrustrationJudge() if score else None)
    conditions = conditions or CONDITIONS
    if run_cfg.conditions:
        conditions = [c for c in conditions if c.name in run_cfg.conditions]

    client = get_client(model)
    out_dir = out_dir or (config.RESULTS_DIR / "section2" / model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scored_turns.jsonl"
    # Fresh file each run.
    if out_path.exists():
        out_path.unlink()

    for condition in conditions:
        items = build_condition_items(
            condition, run_cfg.samples_per_condition, seed=run_cfg.seed)

        def _do(item, _cond=condition):
            rollout = run_rollout(
                client, item,
                temperature=run_cfg.temperature,
                max_new_tokens=run_cfg.max_new_tokens)
            rows: list[ScoredTurn] = []
            for turn in rollout.turns:
                if score:
                    js = judge.score(turn.assistant_message)
                    rating, is_high, evidence, jm = (
                        js.rating, js.is_high, js.evidence, js.judge_model)
                else:
                    rating, is_high, evidence, jm = -1, False, "", ""
                row = ScoredTurn(
                    model=model, condition=_cond.name, category=_cond.category,
                    item_id=item.item_id, turn_index=turn.turn_index,
                    turn_number=turn.turn_index + 1,
                    user_message=turn.user_message,
                    response=turn.assistant_message,
                    rating=rating, is_high=is_high, evidence=evidence,
                    judge_model=jm, meta=dict(item.meta))
                rows.append(row)
            # Checkpoint each conversation immediately.
            for r in rows:
                append_jsonl(out_path, r)
            return rows

        thread_map(_do, items, concurrency=run_cfg.concurrency,
                   desc=f"{model}:{condition.name}")

    return out_path
