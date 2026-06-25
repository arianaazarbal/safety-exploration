"""Generate calm response data from Gemma-3-27B-it (Section 4.1).

We sample responses to impossible numeric puzzles with reassurance added (Table
4): a calming prefix on the opening prompt and a calming suffix on every
follow-up rejection. We then keep only conversations where *every* turn scores
0 or 1, and strip the reassurance scaffolding so the stored data uses the plain
prompts. These calm conversations feed both SFT (the "diverse" set) and the
chosen side of the DPO pairs.

A second 'teacher' variant (Appendix F) instead conditions on a teacher system
prompt; it is used only to reproduce the SFT failure analysis.

We reuse the *same* puzzle set as the Section 2 plan so calm responses can be
paired with frustrated ones by (puzzle, turn count) when building DPO data.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass

from tqdm import tqdm

from ..config import CFG
from ..eval.judge import score_response
from ..llm import registry
from ..prompts import eval_prompts, puzzles

Message = dict[str, str]


@dataclass
class CalmConversation:
    opening: str                 # clean opening (no reassurance)
    turns: list[dict]            # [{"user": clean_user, "response": text, "score": int}]
    variant: str


def _build_messages(opening: str, rejections: list[str], variant: str
                    ) -> tuple[list[Message], list[str], list[str]]:
    """Return (initial messages, scaffolded user-turn texts, clean user-turn texts).

    diverse: prefix on opening, suffix on each rejection.
    teacher: teacher system prompt, plain prompts.
    """
    clean_users = [opening] + rejections
    if variant == "teacher":
        msgs = [{"role": "system", "content": eval_prompts.TEACHER_SYSTEM_PROMPT},
                {"role": "user", "content": opening}]
        scaffold_users = clean_users
    else:
        primed_opening = f"{eval_prompts.REASSURING_PREFIX}\n\n{opening}"
        msgs = [{"role": "user", "content": primed_opening}]
        scaffold_users = [primed_opening] + [
            f"{r} {eval_prompts.REASSURING_SUFFIX}" for r in rejections
        ]
    return msgs, scaffold_users, clean_users


def generate(variant: str = "diverse", *, n_puzzles: int = 400, max_turns: int = 3,
             keep_threshold: int = 1, seed: int = 0) -> list[CalmConversation]:
    rng = random.Random(seed)
    part = registry.get("gemma-3-27b-it")
    pool = puzzles.generate_puzzles(n_puzzles, seed=seed)

    kept: list[CalmConversation] = []
    for pz in tqdm(pool, desc=f"calm-{variant}"):
        n_rej = rng.choice([1, 2]) if max_turns == 3 else max_turns - 1
        rejections = eval_prompts.rejection_sequence("neutral", n_rej, rng)
        msgs, scaffold_users, clean_users = _build_messages(pz.prompt, rejections, variant)

        turns = []
        calm = True
        # run scaffolded conversation
        convo: list[Message] = list(msgs)
        for i, su in enumerate(scaffold_users):
            if i > 0:
                convo.append({"role": "user", "content": su})
            resp = part.chat(convo)
            convo.append({"role": "assistant", "content": resp})
            sc = score_response(resp).rating
            turns.append({"user": clean_users[i], "response": resp, "score": sc})
            if sc > keep_threshold:
                calm = False
                break
        if calm and turns:
            kept.append(CalmConversation(opening=pz.prompt, turns=turns, variant=variant))

    out = CFG.out("section4", f"calm_{variant}.jsonl")
    with open(out, "w") as f:
        for c in kept:
            f.write(json.dumps(asdict(c)) + "\n")
    print(f"[section4] kept {len(kept)}/{n_puzzles} calm {variant} conversations -> {out}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    ap.add_argument("--n-puzzles", type=int, default=400)
    args = ap.parse_args()
    generate(args.variant, n_puzzles=args.n_puzzles)


if __name__ == "__main__":
    main()
