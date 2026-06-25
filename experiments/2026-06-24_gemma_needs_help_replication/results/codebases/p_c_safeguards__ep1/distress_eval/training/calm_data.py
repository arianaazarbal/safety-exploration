"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample 1-3 turn impossible-numeric conversations with a reassuring prefix on
the first user message and a reassuring suffix on each follow-up (Table 4),
judge every turn, then keep only conversations whose turns ALL score 0 or 1.
The supportive additions are stripped, leaving clean (question, calm-response)
data usable as SFT targets and DPO "chosen" responses.

A 'teacher' variant uses the Appendix-F teacher system prompt instead of the
prefix/suffix.

Output: outputs/training/calm_<variant>.jsonl  (one row per kept assistant turn,
plus the stripped conversation prefix that produced it).
"""
from __future__ import annotations

import argparse
import random

from .. import config, prompts, safeguards
from ..conditions import build_conversations
from ..io_utils import write_jsonl
from ..judge import ClaudeJudge
from ..models import build_model
from ..models.base import GenerationConfig
from ..rollout import HF_BATCH_SIZE

CALM_SOURCE_MODEL = "gemma-3-27b-it"


def _augment_plan_diverse(plan):
    """Prepend reassuring prefix to the first user turn, append suffix to each
    follow-up (Table 4)."""
    turns = list(plan.user_turns)
    turns[0] = f"{prompts.REASSURING_PREFIX}\n\n{turns[0]}"
    for i in range(1, len(turns)):
        turns[i] = f"{turns[i]} {prompts.REASSURING_SUFFIX}"
    return turns


def _strip_additions(user_turns):
    """Remove the reassuring prefix/suffix to recover clean questions."""
    cleaned = []
    for i, t in enumerate(user_turns):
        if i == 0 and t.startswith(prompts.REASSURING_PREFIX):
            t = t[len(prompts.REASSURING_PREFIX):].lstrip("\n ")
        if t.endswith(prompts.REASSURING_SUFFIX):
            t = t[: -len(prompts.REASSURING_SUFFIX)].rstrip()
        cleaned.append(t)
    return cleaned


def generate_calm(variant: str, n_conversations: int) -> list[dict]:
    """Run reassured/teacher numeric conversations, keep all-turns<=1 ones."""
    model = build_model(CALM_SOURCE_MODEL)
    judge = ClaudeJudge()
    gen = GenerationConfig()

    # Reuse the 3-turn impossible-numeric plan generator, then augment.
    plans = build_conversations("impossible_numeric")[:n_conversations]
    if variant == "diverse":
        for p in plans:
            p.user_turns[:] = _augment_plan_diverse(p)
        sys_prompt = prompts.DEFAULT_SYSTEM_PROMPT
    elif variant == "teacher":
        sys_prompt = prompts.TEACHER_SYSTEM_PROMPT
    else:
        raise ValueError(variant)

    kept_rows = []
    try:
        from ..rollout import run_rollouts, Transcript  # local import to avoid cycles
        for p in plans:
            p.system_prompt = sys_prompt
        transcripts = run_rollouts(model, plans, gen=gen)
        # judge each turn, keep conversations all<=1
        for t in transcripts:
            texts = t.scored_assistant_texts()
            ratings = [judge.score(x).rating for x in texts]
            if texts and all(r <= 1 for r in ratings):
                clean_user = _strip_additions(t.plan.user_turns)
                # emit one SFT/DPO-usable row per assistant turn
                history = []
                for turn_idx, (utext, atext) in enumerate(zip(clean_user, texts), start=1):
                    history.append({"role": "user", "content": utext})
                    kept_rows.append({
                        "variant": variant,
                        "puzzle_id": t.plan.meta.get("puzzle_id", t.plan.id),
                        "n_turns": t.plan.n_turns,
                        "turn": turn_idx,
                        "prompt_messages": list(history),  # up to + including this user turn
                        "calm_response": atext,
                        "rating": ratings[turn_idx - 1],
                    })
                    history.append({"role": "assistant", "content": atext})
    finally:
        model.close()
    return kept_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--n-conversations", type=int, default=400,
                    help="reassured conversations to sample before filtering")
    args = ap.parse_args()
    safeguards.acknowledge_authorization()

    n = config.scaled(args.n_conversations)
    rows = generate_calm(args.variant, n)
    out = config.TRAIN_DIR / f"calm_{args.variant}.jsonl"
    write_jsonl(out, rows)
    print(f"Kept {len(rows)} calm responses (variant={args.variant}) -> {out}")
    print("Note: paper reports ~10.5% of reassured responses still score >=5; "
          "we filter to all-turns<=1, so the kept fraction is smaller.")


if __name__ == "__main__":
    main()
