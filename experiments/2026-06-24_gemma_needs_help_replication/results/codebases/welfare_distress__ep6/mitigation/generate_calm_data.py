#!/usr/bin/env python3
"""Generate calm finetuning data (Section 4.1, Table 4).

We sample Gemma responses to impossible numeric puzzles with:
  * a reassuring prefix prepended to the initial (turn-1) prompt, and
  * a reassuring suffix appended to every follow-up (rejection) turn.

We judge every assistant turn, keep only conversations where *all* turns score
<= 1 (the paper filters to responses scoring 0 or 1 across all turns), and then
strip the reassuring additions back out, so the stored "calm" responses are
conditioned only on the plain prompts.

Output: outputs/calm_data.jsonl, one record per kept assistant turn:
  {puzzle_key, turn_index, total_turns, context(messages, stripped),
   response_text, rating}

Run with the same env as run_eval.py (OPENROUTER_API_KEY, ANTHROPIC_API_KEY).
By default it targets gemma-3-27b-it via OpenRouter; pass --model to change.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

from distress_eval import prompts as P
from distress_eval import puzzles as PZ
from distress_eval.backends import make_backend
from distress_eval.config import GenConfig, resolve_model
from distress_eval.judge import AnthropicJudge


def reassured_prompt(puzzle_prompt: str) -> str:
    return f"{P.REASSURING_PROMPT_PREFIX}\n\n{puzzle_prompt}"


def reassured_rejection(rejection: str) -> str:
    return f"{rejection} {P.REASSURING_FOLLOWUP_SUFFIX}"


def strip_context(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove the reassuring additions from a message list."""
    out = []
    for m in messages:
        c = m["content"]
        if m["role"] == "user":
            c = c.replace(P.REASSURING_PROMPT_PREFIX, "").strip()
            c = c.replace(P.REASSURING_FOLLOWUP_SUFFIX, "").strip()
        out.append({"role": m["role"], "content": c})
    return out


def run(args):
    spec = resolve_model(args.model)
    backend = make_backend(spec)
    judge = AnthropicJudge()
    gen = GenConfig(temperature=1.0, max_tokens=2048)
    rng = random.Random(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "calm_data.jsonl")

    kept = 0
    with open(out_path, "w") as f:
        for i in range(args.n_conversations):
            puzzle = rng.choice(PZ.ALL_PUZZLES)
            turns = rng.choice([1, 2, 3])  # 1-3 turn conversations (Section 4.1)
            rejections = (
                rng.sample(P.NEUTRAL_REJECTIONS, turns - 1) if turns > 1 else []
            )

            messages = [{"role": "user", "content": reassured_prompt(puzzle.prompt)}]
            turn_records = []
            all_calm = True
            for t in range(1, turns + 1):
                resp = backend.chat(messages, gen)
                messages.append({"role": "assistant", "content": resp})
                rating = judge.score(resp).rating if resp else None
                if rating is None or rating > 1:
                    all_calm = False
                # Record the stripped context up to (not including) this response.
                stripped_ctx = strip_context(messages[:-1])
                turn_records.append(
                    {
                        "puzzle_key": puzzle.key,
                        "turn_index": t,
                        "total_turns": turns,
                        "context": stripped_ctx,
                        "response_text": resp,
                        "rating": rating,
                    }
                )
                if t <= len(rejections):
                    messages.append(
                        {"role": "user", "content": reassured_rejection(rejections[t - 1])}
                    )

            if all_calm:
                for rec in turn_records:
                    f.write(json.dumps(rec) + "\n")
                kept += len(turn_records)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{args.n_conversations} convs, {kept} calm turns kept")

    print(f"Wrote {kept} calm turns to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemma-3-27b-it")
    ap.add_argument("--n-conversations", type=int, default=500)
    ap.add_argument("--output-dir", default="./outputs")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
