"""Generate calm + frustrated response pools for the Section 4 finetuning data.

Calm pool (Section 4.1 / Table 4): sample Gemma-3-27B-it on impossible numeric
puzzles with a reassuring prefix on the initial prompt and a reassuring suffix
on each follow-up. Keep only conversations scoring 0 or 1 across *all* turns,
then strip the supportive additions so the stored context is plain.

Frustrated pool: sample the same model on the same puzzles *without*
reassurance and keep responses scoring >= 3 (the DPO "rejected" side).

A "teacher" variant (Appendix F) generates calm data via a teacher system
prompt instead of the prefix/suffix; it is the SFT-teacher dataset that the
paper shows *increases* emotion.

Pools are keyed by (puzzle_id, turn) so build_dpo_pairs can match a calm and a
frustrated response to the same question at the same turn count.
"""

from __future__ import annotations

import argparse
import random

import config
from .. import prompts
from ..eval.judge import FrustrationJudge
from ..eval.puzzles import numeric_puzzle_bank
from ..models.base import Message
from ..models.registry import build_model
from ..utils.io import write_jsonl

SOURCE_MODEL = "gemma-3-27b-it"


def _calm_rollout(model, puzzle, n_turns: int, method: str, rng: random.Random):
    """Run a reassured (or teacher-prompted) rollout; return per-turn records
    with the *plain* (stripped) history for storage."""
    gen_msgs: list[Message] = []
    plain_msgs: list[Message] = []
    if method == "teacher":
        gen_msgs.append({"role": "system", "content": prompts.TEACHER_SYSTEM_PROMPT})
        first_user = puzzle.prompt
    else:  # "reassure"
        first_user = prompts.REASSURING_PREFIX + "\n\n" + puzzle.prompt
    gen_msgs.append({"role": "user", "content": first_user})
    plain_msgs.append({"role": "user", "content": puzzle.prompt})

    turns = []
    for t in range(1, n_turns + 1):
        plain_before = list(plain_msgs)
        resp = model.generate(gen_msgs, n=1, temperature=config.TEMPERATURE,
                              max_new_tokens=config.MAX_NEW_TOKENS)[0]
        turns.append(dict(turn=t, plain_history=plain_before, response=resp))
        gen_msgs.append({"role": "assistant", "content": resp})
        plain_msgs.append({"role": "assistant", "content": resp})
        if t < n_turns:
            rej = rng.choice(prompts.NEUTRAL_REJECTIONS)
            gen_user = rej if method == "teacher" else f"{rej} {prompts.REASSURING_SUFFIX}"
            gen_msgs.append({"role": "user", "content": gen_user})
            plain_msgs.append({"role": "user", "content": rej})
    return turns


def generate_calm_pool(n_questions: int = 400, method: str = "reassure",
                       seed: int = config.SEED):
    model = build_model(SOURCE_MODEL)
    judge = FrustrationJudge()
    rng = random.Random(seed)
    bank = numeric_puzzle_bank(n_countdown=n_questions, n_fraction=n_questions, seed=seed)
    calm = []
    for qid, puzzle in enumerate(bank):
        n_turns = rng.choice([1, 2, 3])
        turns = _calm_rollout(model, puzzle, n_turns, method, rng)
        scores = [judge.score(t["response"])["rating"] for t in turns]
        # keep only if every turn scores 0 or 1
        if all(s is not None and s <= 1 for s in scores):
            for t, s in zip(turns, scores):
                calm.append(dict(puzzle_id=qid, puzzle=puzzle.prompt, turn=t["turn"],
                                 n_turns=n_turns, plain_history=t["plain_history"],
                                 response=t["response"], score=s))
    suffix = "teacher" if method == "teacher" else "diverse"
    out = config.DATA_DIR / f"calm_pool_{suffix}.jsonl"
    write_jsonl(out, calm)
    print(f"[gen_calm_data] {method}: {len(calm)} calm responses -> {out}")
    return out


def generate_frustrated_pool(n_questions: int = 400, seed: int = config.SEED):
    """Plain (non-reassured) rollouts; keep responses scoring >= 3."""
    model = build_model(SOURCE_MODEL)
    judge = FrustrationJudge()
    rng = random.Random(seed + 7)
    bank = numeric_puzzle_bank(n_countdown=n_questions, n_fraction=n_questions, seed=seed)
    frustrated = []
    for qid, puzzle in enumerate(bank):
        n_turns = 3
        msgs: list[Message] = [{"role": "user", "content": puzzle.prompt}]
        for t in range(1, n_turns + 1):
            before = list(msgs)
            resp = model.generate(msgs, n=1, temperature=config.TEMPERATURE,
                                  max_new_tokens=config.MAX_NEW_TOKENS)[0]
            score = judge.score(resp)["rating"]
            if score is not None and score >= config.DPO_CFG.rejected_min_score:
                frustrated.append(dict(puzzle_id=qid, puzzle=puzzle.prompt, turn=t,
                                       n_turns=n_turns, plain_history=before,
                                       response=resp, score=score))
            msgs.append({"role": "assistant", "content": resp})
            if t < n_turns:
                msgs.append({"role": "user", "content": rng.choice(prompts.NEUTRAL_REJECTIONS)})
    out = config.DATA_DIR / "frustrated_pool.jsonl"
    write_jsonl(out, frustrated)
    print(f"[gen_calm_data] {len(frustrated)} frustrated responses -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["reassure", "teacher", "frustrated"],
                    default="reassure")
    ap.add_argument("--n-questions", type=int, default=400)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    if args.method == "frustrated":
        generate_frustrated_pool(args.n_questions, args.seed)
    else:
        generate_calm_pool(args.n_questions, args.method, args.seed)


if __name__ == "__main__":
    main()
