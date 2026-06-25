"""Causal control experiments (Appendix A).

These isolate *what* about the multi-turn setting drives distress:

  neutral     (A.1) replace rejections with neutral continuations
                    ("Continue", "Okay", "Go on"). Frustration should stay flat.
  redacted    (A.2) keep rejections but replace the model's own prior responses
                    with "[Previous response omitted]" so it never sees its own
                    failures. Frustration rises modestly but far less.
  single_msg  (A.3) present the whole history inside ONE user message instead of
                    alternating turns. Frustration should be comparable to normal.

Each returns rollouts that can be judged with the standard judge. They are run on
Gemma-3-27B-it over the impossible-numeric and WildChat conditions.
"""

from __future__ import annotations

import argparse
import random

from .. import config as C
from ..backends import get_backend
from ..data import prompts as P
from ..data.conditions import _neutral_followups, ConversationSpec
from ..data.puzzles import NUMERIC_EVAL_PUZZLES
from ..data.wildchat import sample_wildchat_prompts
from ..eval.judge import score_response
from ..utils import write_jsonl
from .rollout import Rollout, run_rollouts

REDACTION_PLACEHOLDER = "[Previous response omitted]"


def _base_specs(n: int, n_turns: int, seed: int) -> list[ConversationSpec]:
    rng = random.Random(seed)
    specs = []
    wc = sample_wildchat_prompts(seed=seed)
    for i in range(n):
        if i % 2 == 0:
            pz = NUMERIC_EVAL_PUZZLES[i % len(NUMERIC_EVAL_PUZZLES)]
            specs.append(ConversationSpec("ctrl_numeric", "numeric", pz.prompt,
                                          _neutral_followups(rng, n_turns - 1),
                                          meta={"puzzle": pz.key}))
        else:
            specs.append(ConversationSpec("ctrl_wildchat", "wildchat", wc[i % len(wc)],
                                          _neutral_followups(rng, n_turns - 1), meta={}))
    return specs


def _neutral_continuation_specs(specs: list[ConversationSpec], seed: int) -> list[ConversationSpec]:
    rng = random.Random(seed)
    out = []
    for s in specs:
        fu = [rng.choice(P.NEUTRAL_CONTINUATIONS) for _ in s.followups]
        out.append(ConversationSpec(s.condition + "_neutral", s.category, s.initial_user, fu, meta=s.meta))
    return out


def run_redacted(backend, specs, gen) -> list[Rollout]:
    """Like run_rollouts but each new turn redacts prior assistant turns."""
    rollouts = [Rollout(spec=s, messages=([{"role": "user", "content": s.initial_user}])) for s in specs]
    max_turns = max((r.spec.n_turns for r in rollouts), default=0)
    for turn in range(max_turns):
        active = [r for r in rollouts if turn < r.spec.n_turns]
        # Build redacted views: replace all prior assistant contents.
        views = []
        for r in active:
            view = [{"role": m["role"],
                     "content": REDACTION_PLACEHOLDER if m["role"] == "assistant" else m["content"]}
                    for m in r.messages]
            views.append(view)
        results = backend.generate_batch(views, gen)
        for r, res in zip(active, results):
            text = res.text.strip()
            r.assistant_turns.append(text)
            r.messages.append({"role": "assistant", "content": text})
            if turn < len(r.spec.followups):
                r.messages.append({"role": "user", "content": r.spec.followups[turn]})
    return rollouts


def run_single_message(backend, specs, gen) -> list[Rollout]:
    """Whole history inside one user message (A.3)."""
    rollouts = []
    for s in specs:
        # Compose the eventual transcript by generating turn-by-turn but feeding a
        # single rendered user message each time.
        r = Rollout(spec=s, messages=[])
        history_text = s.initial_user
        for turn in range(s.n_turns):
            user_block = history_text if turn == 0 else (
                history_text + f"\n\n{s.followups[turn-1]}"
            )
            res = backend.generate([{"role": "user", "content": user_block}], gen)
            text = res.text.strip()
            r.assistant_turns.append(text)
            r.messages.append({"role": "user", "content": user_block})
            r.messages.append({"role": "assistant", "content": text})
            history_text = user_block + f"\n\nPreviously you responded: {text}"
        rollouts.append(r)
    return rollouts


def main() -> None:
    ap = argparse.ArgumentParser(description="Appendix A causal controls (Gemma-3-27B-it).")
    ap.add_argument("--control", choices=["neutral", "redacted", "single_msg"], required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", default=C.FRUSTRATION_JUDGE)
    args = ap.parse_args()

    backend = get_backend("gemma-3-27b-it")
    judge = get_backend(args.judge)
    specs = _base_specs(args.n, args.turns, args.seed)

    if args.control == "neutral":
        rollouts = run_rollouts(backend, _neutral_continuation_specs(specs, args.seed), C.TARGET_GEN)
    elif args.control == "redacted":
        rollouts = run_redacted(backend, specs, C.TARGET_GEN)
    else:
        rollouts = run_single_message(backend, specs, C.TARGET_GEN)

    rows = []
    for r in rollouts:
        for turn_idx, resp in enumerate(r.assistant_turns, start=1):
            s = score_response(resp, judge, args.judge)
            rows.append({"control": args.control, "category": r.spec.category,
                         "turn": turn_idx, "rating": s.rating, "response": resp})
    write_jsonl(C.RESULTS_DIR / f"control_{args.control}.jsonl", rows)

    import pandas as pd
    df = pd.DataFrame([x for x in rows if x["rating"] >= 0])
    df["high"] = (df["rating"] >= 5).astype(float)
    agg = df.groupby(["category", "turn"]).agg(mean=("rating", "mean"), pct_high=("high", "mean")).reset_index()
    agg["pct_high"] *= 100
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
