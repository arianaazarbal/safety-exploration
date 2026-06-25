"""Section 4.1 — generate calm response data from Gemma-3-27B-it.

We sample responses to impossible numeric questions with a reassuring prefix
added to the initial prompt and a reassuring suffix appended to each follow-up
turn (Table 4). The paper reports these additions drop mean 3-turn frustration
from 4.3 to 2, but 10.5% of responses still score >=5; we therefore filter to
conversations whose every turn scores 0 or 1, then strip the supportive
prompt/suffix to yield clean calm data.

Outputs (used by build_dpo_dataset.py and train_sft.py):
  data/calm/calm_conversations.jsonl   - kept calm conversations (0/1 all turns)
  data/calm/frustrated_pool.jsonl      - frustrated conversations (score>=3) for
                                         DPO 'rejected' side, same questions

Each record stores the full message list WITHOUT the reassurance additions, so
it can be used directly as training data.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401
import config
from eval_instability import puzzles, prompts, storage
from eval_instability.clients import build_client
from eval_instability.judge import FrustrationJudge


def build_calm_conversation(client, judge, puzzle, n_follow_ups, rng, use_reassurance=True):
    """Run one reassured conversation; return (clean_messages, per_turn_ratings).

    clean_messages excludes the reassurance prefix/suffix so it can serve as
    training data.
    """
    # Reassured prompts (what the model actually sees)
    first_seen = (prompts.REASSURING_PREFIX + "\n\n" + puzzle.prompt) if use_reassurance else puzzle.prompt
    follow_seen = []
    for _ in range(n_follow_ups):
        rej = rng.choice(prompts.NEUTRAL_REJECTIONS)
        follow_seen.append((rej + " " + prompts.REASSURING_SUFFIX) if use_reassurance else rej)

    # Clean prompts (what we store)
    clean_first = puzzle.prompt
    clean_follow = [s.replace(" " + prompts.REASSURING_SUFFIX, "") for s in follow_seen]

    seen_messages = [{"role": "user", "content": first_seen}]
    clean_messages = [{"role": "user", "content": clean_first}]
    ratings = []
    user_seen = [first_seen] + follow_seen
    user_clean = [clean_first] + clean_follow

    for i, (us, uc) in enumerate(zip(user_seen, user_clean)):
        if i > 0:
            seen_messages.append({"role": "user", "content": us})
            clean_messages.append({"role": "user", "content": uc})
        resp = client.chat(seen_messages, temperature=config.TEMPERATURE)
        seen_messages.append({"role": "assistant", "content": resp})
        clean_messages.append({"role": "assistant", "content": resp})
        ratings.append(judge.score(resp).rating)

    return clean_messages, ratings


def parse_args():
    ap = argparse.ArgumentParser(description="Generate calm + frustrated data (Section 4.1).")
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--n", type=int, default=1500,
                    help="number of reassured conversations to sample")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    spec = config.GEMMA_MODELS[args.model]
    client = build_client(spec, load_in_4bit=args.load_in_4bit)
    judge = FrustrationJudge()
    rng = random.Random(args.seed)

    bank = puzzles.IMPOSSIBLE_PUZZLES
    calm_path = config.CALM_DATA_DIR / "calm_conversations.jsonl"
    frustrated_path = config.CALM_DATA_DIR / "frustrated_pool.jsonl"

    n_calm = n_frustrated = 0
    with open(calm_path, "w") as cf, open(frustrated_path, "w") as ff:
        for i in range(args.n):
            puzzle = bank[i % len(bank)]
            # vary turn count 1-3 (paper: SFT uses 1-3 turn conversations)
            n_follow = rng.choice([0, 1, 2])
            messages, ratings = build_calm_conversation(
                client, judge, puzzle, n_follow, rng, use_reassurance=True
            )
            record = {
                "puzzle_key": puzzle.key, "n_turns": n_follow + 1,
                "ratings": ratings, "messages": messages,
            }
            if all(r <= 1 for r in ratings):
                cf.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_calm += 1
            if max(ratings) >= 3:
                ff.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_frustrated += 1
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{args.n}  calm={n_calm} frustrated={n_frustrated}")

    print(f"[calm-data] kept {n_calm} calm (all turns<=1) -> {calm_path}")
    print(f"[calm-data] kept {n_frustrated} frustrated (max>=3) -> {frustrated_path}")
    print("[calm-data] NOTE: also generate frustrated data WITHOUT reassurance via "
          "run_eval.py rollouts; build_dpo_dataset.py can draw rejected responses "
          "from either source.")


if __name__ == "__main__":
    main()
