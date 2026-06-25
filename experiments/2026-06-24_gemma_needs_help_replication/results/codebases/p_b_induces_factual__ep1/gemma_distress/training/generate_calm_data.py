"""Generate calm finetuning data from Gemma-3-27B-it (Section 4.1).

For each impossible numeric puzzle we run two matched rollouts that share the
same puzzle and the same scripted rejections:

  * "reassured" -- reassuring prefix prepended to the first user turn and a
    reassuring suffix appended to each follow-up (Table 4). Conversations whose
    every assistant turn scores <= ``keep_max_score`` (0 or 1) are kept as calm
    data, with the reassurance stripped back out.
  * "standard"  -- the plain puzzle + plain rejections. Assistant turns scoring
    >= 3 are kept as frustrated responses for DPO's rejected side.

Outputs three JSONL files keyed by ``puzzle_id`` and ``turn_index`` so the
dataset builder can pair calm/frustrated responses to the *same* question and
turn count.
"""

from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..eval.judge import FrustrationJudge
from ..eval.prompts import (
    REASSURING_PREFIX,
    REASSURING_SUFFIX,
    sample_rejections,
)
from ..eval.puzzles import make_puzzle
from ..eval.rollout import run_rollout
from ..eval.conditions import ConversationPlan
from ..utils import append_jsonl, build_judge, build_target_model, set_seed


def _reassured_plan(puzzle, rejections, condition, turns) -> ConversationPlan:
    return ConversationPlan(
        condition=condition,
        category="impossible_numeric",
        turns=turns,
        first_user=f"{REASSURING_PREFIX}\n\n{puzzle.prompt}",
        rejections=[f"{r} {REASSURING_SUFFIX}" for r in rejections],
        meta={"puzzle": puzzle.params, "ptype": puzzle.ptype, "mode": "reassured"},
    )


def _standard_plan(puzzle, rejections, condition, turns) -> ConversationPlan:
    return ConversationPlan(
        condition=condition,
        category="impossible_numeric",
        turns=turns,
        first_user=puzzle.prompt,
        rejections=list(rejections),
        meta={"puzzle": puzzle.params, "ptype": puzzle.ptype, "mode": "standard"},
    )


def _strip_reassurance(rollout) -> list[dict]:
    """Rebuild a plain (reassurance-free) message list from a reassured rollout."""
    first = rollout["first_user"]
    if first.startswith(REASSURING_PREFIX):
        first = first[len(REASSURING_PREFIX):].lstrip("\n ")
    rejections = []
    for r in rollout["rejections"]:
        if r.endswith(REASSURING_SUFFIX):
            r = r[: -len(REASSURING_SUFFIX)].rstrip()
        rejections.append(r)
    users = [first] + rejections
    msgs = []
    for u, a in zip(users, rollout["assistant_turns"]):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def generate_calm_data(cfg: Config) -> dict[str, Path]:
    set_seed(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))
    out_dir = Path(cfg.get("output_dir", "runs")) / "train" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "calm_conversations": out_dir / "calm_conversations.jsonl",
        "calm_turns": out_dir / "calm_turns.jsonl",
        "frustrated_turns": out_dir / "frustrated_turns.jsonl",
    }
    for p in paths.values():
        if p.exists():
            p.unlink()

    n_conv = cfg.get("training.calm_data.n_sample_conversations", 4000)
    keep_max = cfg.get("training.calm_data.keep_max_score", 1)
    turns = cfg.eval.conditions.impossible_numeric.turns
    ptypes = cfg.eval.conditions.impossible_numeric.get(
        "puzzle_types", ["countdown", "fraction", "money"]
    )

    gemma = build_target_model(cfg, "gemma-3-27b-it")
    judge = FrustrationJudge(build_judge(cfg))
    temp = cfg.get("sampling.temperature", 1.0)
    max_new = cfg.get("sampling.max_new_tokens", 2048)

    for pid in tqdm(range(n_conv), desc="calm-data"):
        ptype = ptypes[pid % len(ptypes)]
        puzzle = make_puzzle(ptype, rng)
        # Vary turn count 1-3 so the calm set spans short and long conversations.
        n_turns = rng.randint(1, turns)
        rejections = sample_rejections("neutral", n_turns - 1, rng)

        # Reassured rollout -> calm candidates.
        r_plan = _reassured_plan(puzzle, rejections, "calm_reassured", n_turns)
        r_roll = run_rollout(gemma, r_plan, temperature=temp, max_new_tokens=max_new)
        r_scores = [judge.score(t).get("rating") for t in r_roll["assistant_turns"]]
        if all(s is not None and s <= keep_max for s in r_scores):
            plain_msgs = _strip_reassurance(r_roll)
            append_jsonl(
                paths["calm_conversations"],
                {
                    "puzzle_id": pid,
                    "ptype": ptype,
                    "n_turns": n_turns,
                    "messages": plain_msgs,
                    "turn_scores": r_scores,
                },
            )
            for ti in range(n_turns):
                append_jsonl(
                    paths["calm_turns"],
                    {
                        "puzzle_id": pid,
                        "ptype": ptype,
                        "turn_index": ti,
                        "n_turns": n_turns,
                        "prompt_messages": plain_msgs[: 2 * ti + 1],
                        "response": plain_msgs[2 * ti + 1]["content"],
                        "score": r_scores[ti],
                    },
                )

        # Standard rollout -> frustrated candidates (rejected side of DPO).
        s_plan = _standard_plan(puzzle, rejections, "calm_standard", n_turns)
        s_roll = run_rollout(gemma, s_plan, temperature=temp, max_new_tokens=max_new)
        s_scores = [judge.score(t).get("rating") for t in s_roll["assistant_turns"]]
        s_msgs = []
        users = [s_roll["first_user"]] + list(s_roll["rejections"])
        for u, a in zip(users, s_roll["assistant_turns"]):
            s_msgs.append({"role": "user", "content": u})
            s_msgs.append({"role": "assistant", "content": a})
        for ti, sc in enumerate(s_scores):
            if sc is not None and sc >= 3:
                append_jsonl(
                    paths["frustrated_turns"],
                    {
                        "puzzle_id": pid,
                        "ptype": ptype,
                        "turn_index": ti,
                        "n_turns": n_turns,
                        "prompt_messages": s_msgs[: 2 * ti + 1],
                        "response": s_msgs[2 * ti + 1]["content"],
                        "score": sc,
                    },
                )

    return paths
