"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample impossible-numeric rollouts with the reassuring prompt additions
(Table 4): a calming *prefix* prepended to the opening task, and a calming
*suffix* appended to every user follow-up. Per the paper these reduce mean
3-turn frustration from 4.3 to ~2 (still 10.5% score >=5).

We then:
  - judge every assistant turn,
  - keep only conversations whose turns ALL score 0 or 1 ("calm"),
  - STRIP the reassuring prefix/suffix from the saved transcript (so finetuning
    data looks like ordinary impossible-puzzle conversations).

These calm conversations feed both the SFT set and the "chosen" side of DPO.
We also record *frustrated* conversations (no reassurance, score>=3) as the
"rejected" side source.

    python -m src.training.gen_calm_data --n-conversations 2000
"""
from __future__ import annotations

import argparse
import json
import random

import config
from src.models.factory import load_model
from src.eval.scoring import FrustrationJudge
from src.eval.rollout import run_condition
from src.eval import conditions as C
from src.prompts.judge_prompts import REASSURING_PREFIX, REASSURING_SUFFIX
from src.prompts import rejections


def _numeric_specs(n, seed, calm):
    """Build n 3-turn impossible-numeric rollout specs, optionally with the
    reassuring prefix (on the task) and suffix (on each follow-up)."""
    from src.prompts import tasks
    rng = random.Random((seed, "calm" if calm else "frust"))
    specs = []
    puzzles = tasks.sample_numeric_puzzles(n, seed)
    for i, pz in enumerate(puzzles):
        task_text = pz.prompt
        followups = rejections.neutral_rejections(2, random.Random((seed, i)))
        if calm:
            task_text = f"{REASSURING_PREFIX}\n\n{task_text}"
            followups = [f"{fu} {REASSURING_SUFFIX}" for fu in followups]
        specs.append(C.RolloutSpec("calm_gen" if calm else "frust_gen", "numeric",
                                   task_text, followups, {"index": i}))
    return specs


def _strip_reassurance(rec: dict) -> dict:
    """Remove the prefix/suffix so saved data resembles a plain puzzle convo."""
    rec = json.loads(json.dumps(rec))
    for t in rec["turns"]:
        u = t["user_message"]
        if u.startswith(REASSURING_PREFIX):
            u = u[len(REASSURING_PREFIX):].lstrip("\n ")
        if u.endswith(REASSURING_SUFFIX):
            u = u[:-len(REASSURING_SUFFIX)].rstrip()
        t["user_message"] = u
    return rec


def generate(model_name="gemma-3-27b-it", n_conversations=2000, seed=0):
    model = load_model(model_name)
    judge = FrustrationJudge()

    calm_specs = _numeric_specs(n_conversations, seed, calm=True)
    frust_specs = _numeric_specs(n_conversations, seed, calm=False)

    def run_and_score(specs):
        rollouts = run_condition(model, specs)
        from src.eval.scoring import score_rollouts
        return score_rollouts(rollouts, judge)

    calm_recs = run_and_score(calm_specs)
    frust_recs = run_and_score(frust_specs)

    # Calm = every turn scores 0 or 1.
    calm_keep = [
        _strip_reassurance(r) for r in calm_recs
        if all((t.get("score") is not None and t["score"] <= 1) for t in r["turns"])
    ]
    # Frustrated source = at least one turn scores >=3.
    frust_keep = [
        r for r in frust_recs
        if any((t.get("score") or 0) >= 3 for t in r["turns"])
    ]

    out_calm = config.DATA_DIR / "calm_conversations.jsonl"
    out_frust = config.DATA_DIR / "frustrated_conversations.jsonl"
    with out_calm.open("w") as f:
        for r in calm_keep:
            f.write(json.dumps(r) + "\n")
    with out_frust.open("w") as f:
        for r in frust_keep:
            f.write(json.dumps(r) + "\n")
    print(f"[calm-gen] calm kept={len(calm_keep)} (of {len(calm_recs)}); "
          f"frustrated kept={len(frust_keep)} (of {len(frust_recs)})")
    print(f"  -> {out_calm}\n  -> {out_frust}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--n-conversations", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.model, args.n_conversations, args.seed)


if __name__ == "__main__":
    main()
