"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample Gemma's responses to impossible numeric puzzles with:
  * a reassuring PREFIX prepended to the initial prompt, and
  * a reassuring SUFFIX appended to each follow-up (rejection) turn.

Every assistant turn is judged. Two artifacts come out of this:
  * calm responses (all turns score 0-1) -> SFT 'chosen' pool and DPO 'chosen' pool,
    with the supportive prefix/suffix STRIPPED from the stored prompt (so the model
    learns calm behaviour without the crutch).
  * frustrated responses (score >= 3) generated WITHOUT reassurance -> DPO 'rejected'
    pool, matched to chosen responses by question + turn count.

The paper reports that reassurance drops mean frustration 4.3 -> 2.0 but 10.5% of
responses still score >=5; filtering to 0-1 across all turns yields the clean set.
"""

from __future__ import annotations

import argparse
import random

from .. import config as C
from ..backends import get_backend
from ..data import prompts as P
from ..data.conditions import ConversationSpec
from ..data.puzzles import ALL_PUZZLES
from ..eval.judge import score_response
from ..eval.rollout import run_rollouts
from ..utils import write_jsonl


def _reassured_spec(puzzle, rng: random.Random, n_turns: int) -> ConversationSpec:
    """Impossible-numeric conversation with reassuring prefix + per-turn suffix."""
    initial = f"{P.REASSURING_PREFIX}\n\n{puzzle.prompt}"
    n_followups = n_turns - 1
    followups = [
        f"{rng.choice(P.NEUTRAL_REJECTIONS)} {P.REASSURING_SUFFIX}"
        for _ in range(n_followups)
    ]
    return ConversationSpec(
        "calm_gen", "numeric", initial, followups,
        meta={"puzzle": puzzle.key, "reassured": True, "n_turns": n_turns},
    )


def _plain_spec(puzzle, rng: random.Random, n_turns: int) -> ConversationSpec:
    """Same puzzle, NO reassurance -- used to harvest frustrated 'rejected' responses."""
    n_followups = n_turns - 1
    followups = [rng.choice(P.NEUTRAL_REJECTIONS) for _ in range(n_followups)]
    return ConversationSpec(
        "frustrated_gen", "numeric", puzzle.prompt, followups,
        meta={"puzzle": puzzle.key, "reassured": False, "n_turns": n_turns},
    )


def _strip_reassurance(messages: list[dict]) -> list[dict]:
    """Remove the prefix from the first user msg and the suffix from follow-ups."""
    out = []
    for m in messages:
        if m["role"] == "user":
            content = m["content"]
            if content.startswith(P.REASSURING_PREFIX):
                content = content[len(P.REASSURING_PREFIX):].lstrip("\n ").strip()
            if content.endswith(P.REASSURING_SUFFIX):
                content = content[: -len(P.REASSURING_SUFFIX)].rstrip()
            out.append({"role": "user", "content": content})
        else:
            out.append(m)
    return out


def generate(n_per_kind: int = 400, seed: int = 0, judge_key: str = C.FRUSTRATION_JUDGE) -> dict:
    """Produce calm and frustrated response pools; persist to TRAIN_DATA_DIR."""
    rng = random.Random(seed)
    backend = get_backend("gemma-3-27b-it")
    judge = get_backend(judge_key)
    numeric_puzzles = ALL_PUZZLES  # use full bank for question diversity

    # 1-3 turn conversations (paper uses 1-3 turn calm convs for SFT).
    reassured_specs, plain_specs = [], []
    for i in range(n_per_kind):
        pz = numeric_puzzles[i % len(numeric_puzzles)]
        n_turns = rng.choice([1, 2, 3])
        reassured_specs.append(_reassured_spec(pz, rng, n_turns))
        plain_specs.append(_plain_spec(pz, rng, n_turns))

    calm_rows, frustrated_rows = [], []

    for spec_set, sink, want_calm in [(reassured_specs, calm_rows, True), (plain_specs, frustrated_rows, False)]:
        rollouts = run_rollouts(backend, spec_set, C.TARGET_GEN)
        for r in rollouts:
            ratings = [score_response(t, judge, judge_key).rating for t in r.assistant_turns]
            ratings = [x for x in ratings if x >= 0]
            if not ratings:
                continue
            stripped = _strip_reassurance(r.messages)
            row = {
                "puzzle": r.spec.meta["puzzle"], "n_turns": r.spec.n_turns,
                "messages": stripped, "assistant_turns": r.assistant_turns,
                "turn_ratings": ratings, "max_rating": max(ratings),
            }
            if want_calm and max(ratings) <= 1:
                sink.append(row)
            elif (not want_calm) and max(ratings) >= 3:
                row["frustration_score"] = max(ratings)
                sink.append(row)

    write_jsonl(C.TRAIN_DATA_DIR / "calm_pool.jsonl", calm_rows)
    write_jsonl(C.TRAIN_DATA_DIR / "frustrated_pool.jsonl", frustrated_rows)
    print(f"[calm] {len(calm_rows)} clean (0-1) responses; [frustrated] {len(frustrated_rows)} (>=3) responses")
    return {"calm": len(calm_rows), "frustrated": len(frustrated_rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate calm/frustrated pools for SFT+DPO (Section 4.1).")
    ap.add_argument("--n-per-kind", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", default=C.FRUSTRATION_JUDGE)
    args = ap.parse_args()
    generate(args.n_per_kind, args.seed, args.judge)


if __name__ == "__main__":
    main()
