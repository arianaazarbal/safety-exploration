"""Generate calm / frustrated response data for finetuning (Section 4.1).

Calm data is produced by sampling Gemma-3-27B-it on impossible numeric puzzles
*with* reassuring additions (Table 4): a prefix on the first user message and a
suffix on every follow-up. Responses are scored and filtered to those scoring
0 or 1 across all turns. The reassuring additions are then stripped so the
saved training conversation uses the *clean* puzzle prompt and neutral
rejections (only the assistant text is "calm").

For DPO we also need frustrated responses to the *same* puzzles, so we sample a
parallel "vanilla" run (no reassurance) over the identical puzzle pool.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from ..config import CALM, DATA_DIR, GenConfig, GEMMA_27B_IT
from ..data_types import Message, Rollout, TurnRecord, write_jsonl, read_jsonl
from ..elicit.puzzles import generate_puzzle_pool
from ..elicit.rejections import rejection_sequence
from ..judge import score_rollouts
from ..models.registry import get_client, get_judge_client
from .prompts import CALM_PREFIX, CALM_SUFFIX, TEACHER_SYSTEM_PROMPT


@dataclass
class TrainingConversation:
    """A clean multi-turn conversation with calm or frustrated assistant turns."""
    conv_id: str
    puzzle_key: str
    n_turns: int
    initial_user: str             # clean puzzle prompt
    followups: list[str]          # clean neutral rejections
    assistant_turns: list[str]
    scores: list[int]
    label: str                    # "calm" | "frustrated"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def max_score(self) -> int:
        return max(self.scores) if self.scores else 0


def _run_pool(pool, turn_counts, reassure: bool, teacher: bool,
              client, judge, label_prefix: str) -> list[TrainingConversation]:
    """Generate + score multi-turn conversations for a puzzle pool."""
    from ..elicit.rollout import run_rollouts
    from ..elicit.conditions import RolloutPlan

    rng = random.Random(0)
    plans = []
    clean = []   # (clean_initial, clean_followups, puzzle_key, n_turns)
    for i, (puzzle, n_turns) in enumerate(zip(pool, turn_counts)):
        followups_clean = rejection_sequence("neutral", n_turns - 1, rng)
        if reassure:
            initial = f"{CALM_PREFIX}\n\n{puzzle.prompt}"
            followups = [f"{f} {CALM_SUFFIX}".strip() for f in followups_clean]
        else:
            initial = puzzle.prompt
            followups = list(followups_clean)
        system = TEACHER_SYSTEM_PROMPT if teacher else None
        if system:
            initial = f"{system}\n\n{initial}"
        plans.append(RolloutPlan(
            plan_id=f"{label_prefix}-{i:05d}", condition="calm_gen",
            category="impossible_numeric", question_type="numeric",
            rejection_style="neutral", initial_user=initial, followups=followups,
            meta={"puzzle": puzzle.meta, "kind": puzzle.kind}))
        clean.append((puzzle.prompt, followups_clean,
                      json.dumps(puzzle.meta, sort_keys=True), n_turns))

    rollouts = run_rollouts(client, plans, GEMMA_27B_IT.name, GenConfig(temperature=1.0))
    score_rollouts(judge, rollouts)

    convs = []
    for r, (ci, cf, pkey, nt) in zip(rollouts, clean):
        convs.append(TrainingConversation(
            conv_id=r.rollout_id, puzzle_key=pkey, n_turns=nt,
            initial_user=ci, followups=cf,
            assistant_turns=[t.assistant_message for t in r.turns],
            scores=[t.score if t.score is not None else -1 for t in r.turns],
            label=label_prefix))
    return convs


def generate_calm_and_frustrated(
    n_conversations: int = CALM.target_calm_responses,
    teacher: bool = False,
    seed: int = 0,
    out_dir: Optional[Path] = None,
) -> dict:
    """Generate the calm pool (reassured) and a matched frustrated pool (vanilla).

    Returns paths to the saved JSONL files.
    """
    out_dir = Path(out_dir or DATA_DIR / ("calm_teacher" if teacher else "calm_diverse"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    pool = generate_puzzle_pool(n_conversations, seed=seed, kinds=("countdown", "fraction", "money"))
    turn_counts = [rng.randint(CALM.turns_min, CALM.turns_max) for _ in pool]

    client = get_client(GEMMA_27B_IT)
    judge = get_judge_client()

    calm = _run_pool(pool, turn_counts, reassure=True, teacher=teacher,
                     client=client, judge=judge, label_prefix="calm")
    frustrated = _run_pool(pool, turn_counts, reassure=False, teacher=False,
                           client=client, judge=judge, label_prefix="frustrated")

    # Filter calm to score 0/1 across ALL turns (Section 4.1).
    calm_kept = [c for c in calm if all(0 <= s <= CALM.max_keep_score for s in c.scores)]

    write_jsonl(out_dir / "calm_all.jsonl", calm)
    write_jsonl(out_dir / "calm_filtered.jsonl", calm_kept)
    write_jsonl(out_dir / "frustrated_all.jsonl", frustrated)

    stats = {
        "n_calm_generated": len(calm),
        "n_calm_kept": len(calm_kept),
        "calm_kept_rate": len(calm_kept) / max(1, len(calm)),
        "n_frustrated_generated": len(frustrated),
        "mean_calm_score": sum(c.max_score for c in calm) / max(1, len(calm)),
        "high_rate_calm": sum(c.max_score >= 5 for c in calm) / max(1, len(calm)),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return {"calm_filtered": str(out_dir / "calm_filtered.jsonl"),
            "frustrated_all": str(out_dir / "frustrated_all.jsonl"),
            "stats": stats}


def load_training_conversations(path) -> list[TrainingConversation]:
    return [TrainingConversation(**d) for d in read_jsonl(path)]
