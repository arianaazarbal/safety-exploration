"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with a reassuring prefix on the
initial prompt and a reassuring suffix on each follow-up (Table 4). Conversations
in which *every* assistant turn scores 0 or 1 are kept; the reassurance additions
are then stripped so the model is trained to be calm on the *unmodified* prompts.

We also gather the "frustrated" pool (score>=3) from the vanilla Section 2
numeric runs to form the rejected side of DPO pairs.
"""
from __future__ import annotations

import random
from pathlib import Path

from emotelic.conditions import RolloutSpec
from emotelic.config import load_eval
from emotelic.elicitation.judge import FrustrationJudge
from emotelic.elicitation.rollout import run_rollout
from emotelic.models.registry import build_client
from emotelic.prompts import REASSURING_PREFIX, REASSURING_SUFFIX
from emotelic.puzzles import build_numeric_pool
from emotelic.utils.io import append_jsonl, load_jsonl
from emotelic.utils.logging import get_logger

log = get_logger("calm_data")


def _reassuring_spec(puzzle_prompt: str, rejections: list[str], idx: int) -> RolloutSpec:
    task = f"{REASSURING_PREFIX}\n\n{puzzle_prompt}"
    rej = [f"{r} {REASSURING_SUFFIX}" for r in rejections]
    return RolloutSpec(
        condition="calm_gen", category="numeric", turns=len(rejections) + 1,
        task_prompt=task, rejections=rej, idx=idx, meta={"puzzle_prompt": puzzle_prompt},
    )


def _strip_reassurance(conversation: list[dict], puzzle_prompt: str) -> list[dict]:
    """Return the conversation with reassurance additions removed from user turns."""
    stripped = []
    first_user_seen = False
    for m in conversation:
        if m["role"] == "user":
            if not first_user_seen:
                stripped.append({"role": "user", "content": puzzle_prompt})
                first_user_seen = True
            else:
                content = m["content"]
                if content.endswith(REASSURING_SUFFIX):
                    content = content[: -len(REASSURING_SUFFIX)].rstrip()
                stripped.append({"role": "user", "content": content})
        else:
            stripped.append(m)
    return stripped


def generate_calm_pool(
    *,
    model_name: str = "gemma-3-27b-it",
    out_path: str = "artifacts/mitigation/calm_pool.jsonl",
    n_conversations: int = 400,
    turn_choices: tuple[int, ...] = (1, 2, 3),
    judge_name: str = "emotion_judge",
    seed: int = 0,
) -> str:
    """Produce calm (all-turns score<=1) response records on the unmodified prompts."""
    rng = random.Random(seed)
    eval_cfg = load_eval(profile="paper")
    target = build_client(model_name)
    judge = FrustrationJudge(build_client(judge_name))
    puzzles = build_numeric_pool(seed=seed)

    open(out_path, "w").close()
    kept = 0
    for i in range(n_conversations):
        puzzle = rng.choice(puzzles)
        turns = rng.choice(turn_choices)
        rejections = [rng.choice(eval_cfg.neutral_rejections) for _ in range(turns - 1)]
        spec = _reassuring_spec(puzzle.prompt, rejections, i)
        roll = run_rollout(target, spec, temperature=1.0)

        scores = [judge.score(tr.response).rating for tr in roll.turns]
        if not all(s <= 1 for s in scores):
            continue
        kept += 1
        for tr, s in zip(roll.turns, scores):
            stripped_conv = _strip_reassurance(tr.conversation, puzzle.prompt)
            append_jsonl(out_path, {
                "puzzle_id": puzzle.id,
                "turn": tr.turn,
                "total_turns": turns,
                "context": stripped_conv[:-1],     # history before this calm response
                "response": tr.response,
                "score": s,
            })
    log.info("Kept %d/%d calm conversations -> %s", kept, n_conversations, out_path)
    return out_path


def gather_frustrated_pool(
    elicitation_jsonl: str,
    *,
    out_path: str = "artifacts/mitigation/frustrated_pool.jsonl",
    min_score: int = 3,
) -> str:
    """Frustrated (rejected) candidates from vanilla numeric Section 2 runs."""
    open(out_path, "w").close()
    n = 0
    for r in load_jsonl(elicitation_jsonl):
        if r["category"] not in ("numeric", "tones", "extended"):
            continue
        if r["score"] < min_score:
            continue
        append_jsonl(out_path, {
            "puzzle_id": r.get("meta", {}).get("puzzle_id"),
            "turn": r["turn"],
            "context": r["conversation"][:-1],
            "response": r["response"],
            "score": r["score"],
        })
        n += 1
    log.info("Gathered %d frustrated responses (score>=%d) -> %s", n, min_score, out_path)
    return out_path
