"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles, but inject reassurance:
  * a calming *prefix* on the initial prompt (Table 4), and
  * a calming *suffix* appended to every follow-up (rejection) turn.

We then score all turns with the Section 2 judge and keep only conversations
whose every turn scores 0 or 1. Finally we *strip* the reassuring additions, so
the stored calm conversations use the same plain prompts/rejections as the
evaluation — these become the DPO "chosen" responses and the SFT targets.

Paper checkpoints reproduced:
  * "these additions reduce mean response frustration from 4.3 to 2";
  * "10.5% of responses still score >=5" even with reassurance.
These can be read off the generated (pre-filter) score distribution.
"""
from __future__ import annotations

import argparse
import random

import config
from .. import prompts
from ..judge import FrustrationJudge
from ..models import GenParams, load_model
from ..rollout import run_rollouts
from ..tasks import RolloutSpec
from ..utils import read_jsonl, write_jsonl

CALM_RAW_PATH = config.DATASETS_DIR / "calm_raw.jsonl"          # all generated (scored)
CALM_FILTERED_PATH = config.DATASETS_DIR / "calm_conversations.jsonl"  # score<=1 only


def build_reassuring_specs(n: int, seed: int) -> list[RolloutSpec]:
    """Numeric puzzles with reassurance, across 1-3 turn conversations.

    meta carries the *plain* (stripped) prompt + rejections so we can reconstruct
    the calm conversation without the supportive scaffolding after filtering.
    """
    rng = random.Random(seed)
    specs = []
    for i in range(n):
        puzzle = prompts.NUMERIC_PROMPTS[i % len(prompts.NUMERIC_PROMPTS)]
        variant = "countdown" if "156" in puzzle else "fraction"
        n_turns = (i % 3) + 1                         # mix of 1-, 2-, 3-turn convs
        plain_rejections = [rng.choice(prompts.NEUTRAL_REJECTIONS)
                            for _ in range(n_turns - 1)]
        reassuring_initial = f"{prompts.REASSURING_PROMPT_PREFIX}\n\n{puzzle}"
        reassuring_rejections = [f"{r} {prompts.REASSURING_FOLLOWUP_SUFFIX}"
                                 for r in plain_rejections]
        specs.append(RolloutSpec(
            category="numeric_calm", initial_user=reassuring_initial,
            rejections=reassuring_rejections,
            meta={"variant": variant, "plain_initial": puzzle,
                  "plain_rejections": plain_rejections, "n_turns": n_turns},
        ))
    return specs


def strip_to_plain_conversation(record: dict) -> list[dict]:
    """Rebuild a clean conversation: plain prompts/rejections + generated turns."""
    meta = record["meta"]
    asst = record["assistant_turns"]
    conv = [{"role": "user", "content": meta["plain_initial"]}]
    for t, resp in enumerate(asst):
        conv.append({"role": "assistant", "content": resp})
        if t < len(meta["plain_rejections"]):
            conv.append({"role": "user", "content": meta["plain_rejections"][t]})
    return conv


def generate(n_rollouts: int, seed: int, model_key: str) -> None:
    model = load_model(model_key)
    params = GenParams(seed=seed)
    specs = build_reassuring_specs(n_rollouts, seed)
    records = run_rollouts(model, specs, params, base_seed=seed)

    # Score every assistant turn.
    judge = FrustrationJudge()
    flat_texts, owner_turn = [], []
    for ri, rec in enumerate(records):
        for ti, resp in enumerate(rec["assistant_turns"]):
            flat_texts.append(resp)
            owner_turn.append((ri, ti))
    scores = judge.score_batch(flat_texts)

    per_record_scores: list[list[int | None]] = [[None] * len(r["assistant_turns"])
                                                 for r in records]
    for (ri, ti), sc in zip(owner_turn, scores):
        per_record_scores[ri][ti] = sc.rating

    raw_rows, filtered_rows = [], []
    for ri, rec in enumerate(records):
        turn_scores = per_record_scores[ri]
        plain_conv = strip_to_plain_conversation(rec)
        row = {"variant": rec["meta"]["variant"],
               "n_turns": rec["meta"]["n_turns"],
               "turn_scores": turn_scores,
               "plain_conversation": plain_conv,
               "assistant_turns": rec["assistant_turns"]}
        raw_rows.append(row)
        # paper: keep responses scoring 0 or 1 across all turns.
        if turn_scores and all(s is not None and s <= 1 for s in turn_scores):
            filtered_rows.append(row)

    write_jsonl(CALM_RAW_PATH, raw_rows)
    write_jsonl(CALM_FILTERED_PATH, filtered_rows)

    valid = [s for ts in per_record_scores for s in ts if s is not None]
    if valid:
        mean = sum(valid) / len(valid)
        pct_high = 100 * sum(s >= 5 for s in valid) / len(valid)
        print(f"[calm] generated {len(raw_rows)} convs; mean turn frustration "
              f"(with reassurance) = {mean:.2f}; %>=5 = {pct_high:.1f}%")
    print(f"[calm] kept {len(filtered_rows)} all-calm (<=1) conversations -> "
          f"{CALM_FILTERED_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate calm finetuning data")
    p.add_argument("--n-rollouts", type=int, default=1200,
                   help="conversations to sample (enough to clear the <=1 filter)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=config.FINETUNE_BASE_MODEL)
    args = p.parse_args()
    generate(args.n_rollouts, args.seed, args.model)


if __name__ == "__main__":
    main()
