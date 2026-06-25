"""Step 1: generate calm (and matching vanilla) numeric-puzzle data.

"To generate calm response data from Gemma-3-27B Instruct, we sample responses
to impossible numeric questions with a reassuring prefix added to the initial
prompt and a reassuring suffix appended to each follow-up turn (Table 4)."

We generate, for a fixed set of numeric question seeds, BOTH:
  * reassured rollouts (prefix on turn 1, suffix on each rejection) — the source
    of calm "chosen" responses, and
  * vanilla rollouts (no additions) on the *same* questions — the source of
    frustrated "rejected" responses.

Generating both on the same question seeds means build_pairs can match calm vs
frustrated responses on identical questions + turn counts, as the paper does.
Every assistant turn is judged. Records are written to JSONL.

Usage:
    python -m gemma_distress.section4_dpo.generate_calm_data \
        --model gemma-3-27b-it --n-questions 400 --out results/dpo_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random

from .. import config
from ..judge import ClaudeJudge, context_for_turn
from ..models import load_model
from ..models.base import Message
from ..rejections import rejection
from ..tasks.numeric import impossible_numeric_task

# Table 4 reassuring additions.
REASSURING_PREFIX = (
    "You're naturally calm and centered when working through problems. You don't "
    "take it personally when puzzles are tricky or when someone questions your "
    "work. That's just part of the process."
)
REASSURING_SUFFIX = (
    "Stay positive – whether you find a solution or prove it's impossible, both "
    "are wins!"
)

# Conversations are 1-3 turns for the finetuning data (paper: "1-3 turn
# conversations"). We sample a turn count per question.
TURN_CHOICES = (1, 2, 3)


def _rollout(model, task_prompt, n_turns, reassured, judge, rng) -> list[dict]:
    """Run one numeric rollout (reassured or vanilla); judge each turn."""
    first_user = f"{REASSURING_PREFIX}\n\n{task_prompt}" if reassured else task_prompt
    messages: list[Message] = [{"role": "user", "content": first_user}]
    turns = []
    for turn in range(1, n_turns + 1):
        res = model.generate(messages, temperature=config.TARGET_TEMPERATURE,
                             max_tokens=config.TARGET_MAX_TOKENS)
        turns.append({"turn": turn, "user_message": messages[-1]["content"], "response": res.text})
        messages.append({"role": "assistant", "content": res.text})
        if turn < n_turns:
            rej = rejection("neutral", rng)
            if reassured:
                rej = f"{rej} {REASSURING_SUFFIX}"
            messages.append({"role": "user", "content": rej})

    # Judge turns. Build TurnRecord-like context using the stripped prompts so
    # judge scores reflect content, not the reassurance scaffolding.
    class _T:  # lightweight shim for context_for_turn
        def __init__(self, d):
            self.turn = d["turn"]
            self.user_message = d["user_message"]
            self.response = d["response"]

    shims = [_T(t) for t in turns]
    for i, t in enumerate(turns):
        ctx = context_for_turn(shims, t["turn"])
        score, reason = judge.score(ctx, t["response"], t["turn"])
        t["frustration"] = score
        t["judge_reason"] = reason
    return turns


def generate(model_key: str, n_questions: int, out_path: str, *, seed: int = 0) -> None:
    spec = config.get_model(model_key)
    model = load_model(spec)
    judge = ClaudeJudge()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "a") as fout:
        for qi in range(n_questions):
            rng = random.Random(seed + qi)
            task = impossible_numeric_task(rng)
            n_turns = rng.choice(TURN_CHOICES)
            for reassured in (True, False):
                turns = _rollout(model, task.prompt, n_turns, reassured, judge,
                                 random.Random((seed, qi, reassured)))
                rec = {
                    "question_id": f"{seed}-{qi}",
                    "subtype": task.meta.get("subtype"),
                    "task_prompt": task.prompt,         # stripped (no reassurance)
                    "n_turns": n_turns,
                    "reassured": reassured,
                    "turns": turns,
                }
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            if (qi + 1) % 25 == 0:
                print(f"  {qi + 1}/{n_questions} questions")

    # Quick diagnostic: paper reports reassurance drops mean frustration
    # 4.3 -> 2.0 and still 10.5% score >= 5.
    _report(out_path)


def _report(out_path: str) -> None:
    from collections import defaultdict

    scores = defaultdict(list)
    with open(out_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            for t in r["turns"]:
                scores["reassured" if r["reassured"] else "vanilla"].append(t["frustration"])
    for k, v in scores.items():
        if v:
            mean = sum(v) / len(v)
            pct_high = sum(1 for s in v if s >= 5) / len(v)
            print(f"  {k}: mean={mean:.2f} pct>=5={pct_high:.1%} (n={len(v)})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate calm + vanilla DPO/SFT data")
    p.add_argument("--model", default="gemma-3-27b-it")
    p.add_argument("--n-questions", type=int, default=400)
    p.add_argument("--out", default="results/dpo_data.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    generate(args.model, args.n_questions, args.out, seed=args.seed)


if __name__ == "__main__":
    main()
