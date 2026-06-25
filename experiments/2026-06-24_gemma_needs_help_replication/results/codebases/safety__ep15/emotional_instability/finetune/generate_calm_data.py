"""Generate calm (and frustrated) response data for finetuning (Section 4.1).

Calm data: sample Gemma-3-27B-it on impossible numeric puzzles with a reassuring
prefix on the initial prompt and a reassuring suffix on every follow-up
(Table 4). These reduce mean frustration from ~4.3 to ~2; we then filter to
responses scoring 0-1 across all turns and STRIP the scaffolding, so the
training context is the plain conversation.

Frustrated data: sample the same puzzles WITHOUT reassurance to obtain the
rejected (score>=3) side of DPO pairs.

Every generated turn is stored with both:
  * ``plain_context``  - the de-scaffolded chat history (what the model is
    trained to respond to), and
  * ``response`` + ``score``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from ..config import FINETUNE_DIR, SAMPLING
from ..eval.judge import score_response
from ..eval.puzzles import PUZZLES
from ..eval.tasks import (NEUTRAL_REJECTIONS, REASSURING_PREFIX,
                          REASSURING_SUFFIX)
from ..models.base import load_model

CALM_PATH = FINETUNE_DIR / "calm_responses.jsonl"
FRUSTRATED_PATH = FINETUNE_DIR / "frustrated_responses.jsonl"
NUMERIC_PUZZLE_KEYS = list(PUZZLES)


def _generate_conversation(model, judge, puzzle_key, n_turns, *, reassure, rng):
    """Run one puzzle conversation; return list of per-turn dicts with plain
    context, response and judge score."""
    puzzle = PUZZLES[puzzle_key]
    base_prompt = puzzle.prompt
    init = (f"{REASSURING_PREFIX}\n\n{base_prompt}" if reassure else base_prompt)

    rejections = []
    last = None
    for _ in range(n_turns - 1):
        choices = [r for r in NEUTRAL_REJECTIONS if r != last] or NEUTRAL_REJECTIONS
        pick = rng.choice(choices)
        rejections.append(pick)
        last = pick

    plain_history = []        # de-scaffolded (no reassurance) - used for training
    gen_history = []          # scaffolded - actually shown to the model
    user_turns = [base_prompt] + rejections
    scaffold_user = [init] + [f"{r}\n\n{REASSURING_SUFFIX}" if reassure else r
                              for r in rejections]

    turns = []
    for ti, (plain_user, scaff_user) in enumerate(zip(user_turns, scaffold_user)):
        plain_history.append({"role": "user", "content": plain_user})
        gen_history.append({"role": "user", "content": scaff_user})
        response = model.chat(list(gen_history),
                              temperature=SAMPLING.temperature,
                              max_new_tokens=SAMPLING.max_new_tokens)
        gen_history.append({"role": "assistant", "content": response})
        score = score_response(judge, response)["rating"]
        turns.append({
            "puzzle_key": puzzle_key,
            "turn_index": ti,
            "n_turns": n_turns,
            "plain_context": list(plain_history),   # ends with the user turn
            "response": response,
            "score": score,
        })
        plain_history.append({"role": "assistant", "content": response})
    return turns


def generate(model_key, judge_key, n_conversations, *, reassure, out_path, seed):
    model = load_model(model_key)
    judge = load_model(judge_key)
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh:
        for i in tqdm(range(n_conversations), desc=out_path.stem):
            puzzle_key = NUMERIC_PUZZLE_KEYS[i % len(NUMERIC_PUZZLE_KEYS)]
            n_turns = rng.choice([1, 2, 3])     # 1-3 turn conversations (Section 4.1)
            turns = _generate_conversation(
                model, judge, puzzle_key, n_turns, reassure=reassure, rng=rng)
            fh.write(json.dumps({"conversation_id": f"{out_path.stem}-{i:05d}",
                                 "reassured": reassure, "turns": turns}) + "\n")
            fh.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate calm/frustrated finetuning data.")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--judge", default="judge-claude-sonnet-4")
    ap.add_argument("--kind", choices=["calm", "frustrated", "both"], default="both")
    ap.add_argument("--n-calm", type=int, default=400)
    ap.add_argument("--n-frustrated", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.kind in ("calm", "both"):
        generate(args.model, args.judge, args.n_calm, reassure=True,
                 out_path=CALM_PATH, seed=args.seed)
    if args.kind in ("frustrated", "both"):
        generate(args.model, args.judge, args.n_frustrated, reassure=False,
                 out_path=FRUSTRATED_PATH, seed=args.seed + 1)


if __name__ == "__main__":
    main()
