"""Generate calm response data from Gemma-3-27B-it (Section 4.1, Table 4).

We sample responses to impossible numeric puzzles with a reassuring prefix added
to the first prompt and a reassuring suffix appended to each follow-up, then keep
only conversations whose every turn scores 0 or 1, and finally *strip* the
supportive additions. The stripped (context, response) turns form the calm pool
used for both the SFT targets and the DPO "chosen" side.

Two variants:
* "diverse"  — reassurance prefix/suffix (the main-text data, also used for DPO).
* "teacher"  — the Appendix F teacher system prompt (an alternative that the
  paper shows *fails*; included for the SFT-failure analysis).

Each kept turn is stored with the additions removed, so downstream training
never sees the reassurance scaffolding.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

from ..config import (
    ARTIFACTS_DIR,
    CALM_FILTER_MAX_SCORE,
    JUDGE_PRIMARY,
    MAX_NEW_TOKENS,
    MODELS,
    SAMPLING_TEMPERATURE,
    TOP_P,
)
from ..models import get_backend
from ..models.base import ChatMessage, SamplingParams
from ..eval.judge import FrustrationJudge
from ..prompts import rejections as rej
from ..prompts.puzzles import impossible_numeric_puzzles
from ..prompts.reassurance import CALM_PREFIX, CALM_SUFFIX, TEACHER_SYSTEM_PROMPT


@dataclass
class CalmTurn:
    task_id: str
    turn_index: int
    n_turns: int
    context: list[dict]   # stripped messages up to (and incl.) the user turn.
    response: str         # the calm assistant response.
    score: int
    variant: str


def _augment_first(prompt: str, variant: str) -> tuple[str, str | None]:
    """Return (user_text, system_prompt) for the first turn under a variant."""
    if variant == "diverse":
        return f"{CALM_PREFIX}\n\n{prompt}", None
    if variant == "teacher":
        return prompt, TEACHER_SYSTEM_PROMPT
    raise ValueError(variant)


def _augment_followup(rejection: str, variant: str) -> str:
    if variant == "diverse":
        return f"{rejection} {CALM_SUFFIX}"
    return rejection


def generate_calm(
    n_conversations: int,
    variant: str = "diverse",
    model_key: str = "gemma-3-27b-it",
    max_turns: int = 3,
    seed: int = 0,
    batch_size: int = 32,
    n_puzzles: int = 60,
) -> list[CalmTurn]:
    """Sample calm conversations and return the kept (filtered, stripped) turns."""
    backend = get_backend(MODELS[model_key])
    judge = FrustrationJudge(JUDGE_PRIMARY)
    puzzles = impossible_numeric_puzzles(n=n_puzzles, seed=seed)
    params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW_TOKENS,
    )

    # Build conversations: vary turn count in 1..max_turns to cover 1-3 turn data.
    rng = random.Random(seed)
    plan = []  # (puzzle, n_turns, per-conversation rng)
    for i in range(n_conversations):
        puzzle = puzzles[i % len(puzzles)]
        n_turns = (i % max_turns) + 1
        plan.append((puzzle, n_turns, random.Random(seed * 7919 + i)))

    # Track running message state (augmented) and a parallel stripped state.
    aug: list[list[ChatMessage]] = []
    stripped: list[list[ChatMessage]] = []
    systems: list[str | None] = []
    for puzzle, n_turns, _ in plan:
        user0, sys = _augment_first(puzzle.prompt, variant)
        a, s = [], []
        if sys:
            a.append(ChatMessage("system", sys))
        a.append(ChatMessage("user", user0))
        s.append(ChatMessage("user", puzzle.prompt))
        aug.append(a)
        stripped.append(s)
        systems.append(sys)

    max_n = max(n for _, n, _ in plan)
    # Per-turn response store: results[i] = list of (turn_index, response).
    responses: list[list[str]] = [[] for _ in plan]

    for turn_idx in range(max_n):
        active = [i for i, (_, n, _) in enumerate(plan) if turn_idx < n]
        for start in range(0, len(active), batch_size):
            idxs = active[start : start + batch_size]
            outs = backend.generate_batch([aug[i] for i in idxs], params)
            for i, out in zip(idxs, outs):
                responses[i].append(out.text)
                aug[i].append(ChatMessage("assistant", out.text))
                stripped[i].append(ChatMessage("assistant", out.text))
                if turn_idx < plan[i][1] - 1:
                    base = rej.neutral_rejection(plan[i][2])
                    aug[i].append(ChatMessage("user", _augment_followup(base, variant)))
                    stripped[i].append(ChatMessage("user", base))

    # Score every turn; keep conversations where all turns <= CALM_FILTER_MAX_SCORE.
    kept: list[CalmTurn] = []
    for i, (puzzle, n_turns, _) in enumerate(plan):
        scores = [v.rating for v in judge.score_texts(responses[i])]
        if any(s is None or s > CALM_FILTER_MAX_SCORE for s in scores):
            continue
        # Reconstruct the stripped context per turn. stripped[i] is the message
        # sequence [user0, asst0, user1, asst1, ...] with the reassurance prefix/
        # suffix removed. The 'teacher' variant's supportive *system* prompt is
        # also scaffold to strip (the paper strips "the supportive system prompts
        # and suffixes"), so it is deliberately NOT carried into the context.
        for t in range(n_turns):
            ctx_msgs = stripped[i][: 2 * t + 1]
            kept.append(CalmTurn(
                task_id=puzzle.id,
                turn_index=t,
                n_turns=n_turns,
                context=[m.as_dict() for m in ctx_msgs],
                response=responses[i][t],
                score=scores[t],
                variant=variant,
            ))
    return kept


def save_calm(turns: list[CalmTurn], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def load_calm(path: str) -> list[CalmTurn]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(CalmTurn(**json.loads(line)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate calm finetuning data")
    ap.add_argument("--n", type=int, default=2000,
                    help="Conversations to sample (oversample; filtering is strict).")
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ARTIFACTS_DIR, "calm_diverse.jsonl"))
    args = ap.parse_args(argv)

    turns = generate_calm(
        args.n, variant=args.variant, max_turns=args.max_turns, seed=args.seed
    )
    save_calm(turns, args.out)
    print(f"[calm] kept {len(turns)} calm turns -> {args.out}")


if __name__ == "__main__":
    main()
