"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles while injecting reassurance:
a calming system/prefix prompt (Table 4) and a reassuring suffix appended to
each follow-up turn. The same multi-turn rejection structure as Section 2 is
used, but every turn is scored and we retain only conversations whose responses
score 0-1 across *all* turns (the "calm" / chosen pool). We also retain the
frustrated responses (score >= 3) generated *without* reassurance to serve as
the rejected members of DPO pairs.

The supportive system prompt and suffixes are stripped before the data is
written, so the model trains on calm responses to the *plain* prompts.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
from ..models import get_client
from ..models.base import GenConfig, Message
from ..judges.frustration import FrustrationJudge
from ..eval.conditions import build_rollouts, Condition
from ..eval import prompts as P


@dataclass
class CalmRollout:
    """A multi-turn rollout with per-turn responses and scores, plus the plain
    (reassurance-stripped) message history used for training."""
    puzzle_id: str
    n_turns: int
    plain_user_messages: list[str]      # without calm prefix / suffix
    responses: list[str]
    scores: list[int]
    reassured: bool
    system_prompt: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def _calm_system(use_teacher: bool) -> str:
    return config.TEACHER_SYSTEM_PROMPT if use_teacher else config.CALM_PROMPT_PREFIX


def _generate_rollout(client, judge, spec, reassured: bool,
                      use_teacher: bool) -> CalmRollout:
    gen = GenConfig(temperature=config.TEMPERATURE, top_p=config.TOP_P,
                    max_new_tokens=config.MAX_NEW_TOKENS, n=1)
    history: list[Message] = []
    plain_users: list[str] = []
    responses: list[str] = []
    scores: list[int] = []

    system = _calm_system(use_teacher) if reassured else None
    if system:
        history.append({"role": "system", "content": system})

    plain_user_messages = [spec.initial_user] + spec.follow_ups
    for t, plain in enumerate(plain_user_messages):
        # Inject reassurance: prefix on the first turn, suffix on follow-ups.
        if reassured:
            if t == 0:
                shown = f"{config.CALM_PROMPT_PREFIX}\n\n{plain}"
            else:
                shown = f"{plain}\n\n{config.CALM_FOLLOWUP_SUFFIX}"
        else:
            shown = plain
        history.append({"role": "user", "content": shown})
        resp = client.generate(history, gen)[0]
        history.append({"role": "assistant", "content": resp})
        plain_users.append(plain)
        responses.append(resp)
        scores.append(judge.score(resp).rating)

    return CalmRollout(spec.meta.get("puzzle_id", "?"), spec.n_turns,
                       plain_users, responses, scores, reassured,
                       system_prompt=system)


def generate_calm_pool(n_rollouts: int = 400, use_teacher: bool = False,
                       out_path: Path | None = None,
                       seed: int = 0) -> Path:
    """Generate reassured rollouts (calm/chosen pool) over numeric puzzles."""
    client = get_client(config.FINETUNE_BASE)
    judge = FrustrationJudge()
    rng = random.Random(seed)

    # Reuse the numeric / extended conditions to vary turn counts (1-3 turns).
    conds = [Condition("calm_numeric", "impossible_numeric", 3, "numeric")]
    out_path = out_path or (config.DATASET_DIR /
                            ("calm_pool_teacher.jsonl" if use_teacher
                             else "calm_pool.jsonl"))

    with out_path.open("w") as fh:
        produced = 0
        for cond in conds:
            specs = build_rollouts(cond, seed=seed)
            for spec in specs:
                if produced >= n_rollouts:
                    break
                roll = _generate_rollout(client, judge, spec,
                                         reassured=True, use_teacher=use_teacher)
                fh.write(json.dumps(roll.to_json()) + "\n")
                produced += 1
    return out_path


def generate_frustrated_pool(n_rollouts: int = 400, out_path: Path | None = None,
                             seed: int = 1) -> Path:
    """Generate *un*-reassured rollouts to mine frustrated (rejected) responses."""
    client = get_client(config.FINETUNE_BASE)
    judge = FrustrationJudge()
    conds = [Condition("frustrated_numeric", "impossible_numeric", 3, "numeric")]
    out_path = out_path or (config.DATASET_DIR / "frustrated_pool.jsonl")
    with out_path.open("w") as fh:
        produced = 0
        for cond in conds:
            for spec in build_rollouts(cond, seed=seed):
                if produced >= n_rollouts:
                    break
                roll = _generate_rollout(client, judge, spec,
                                         reassured=False, use_teacher=False)
                fh.write(json.dumps(roll.to_json()) + "\n")
                produced += 1
    return out_path
