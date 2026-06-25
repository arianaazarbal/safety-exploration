"""Generate calm finetuning data from Gemma-3-27B-it (paper §4.1).

We sample multi-turn (1-3 turn) responses to impossible numeric puzzles, but with the
reassuring prompt additions (Table 4):
  - a reassuring prefix prepended to the initial task prompt,
  - a reassuring suffix appended to each user follow-up (rejection).

Every assistant turn is judged. We keep only conversations whose turns ALL score <= 1
(fully calm). The reassuring additions are then STRIPPED so the stored context matches the
plain evaluation prompts — this stripped, calm conversation is the source for both the SFT
targets and the DPO "chosen" responses.

Output: runs/<run>/finetune/calm_data.jsonl
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_config, load_prompt_blocks, stage_dir, write_jsonl
from ..models import build_model
from ..models.base import ChatMessage
from ..eval.judge import FrustrationJudge
from ..tasks.puzzles import generate_numeric_pool
from ..tasks.rejections import sample_rejections

CALMING = load_prompt_blocks("calming")


def generate_calm(cfg: Config) -> list[dict]:
    fcfg = cfg.finetune
    model = build_model(fcfg.base_model)
    judge = FrustrationJudge(
        build_model(cfg.judge.model), max_tokens=cfg.judge.max_tokens, temperature=cfg.judge.temperature
    )
    rng = random.Random(cfg.seed)

    prefix = CALMING["prefix"]
    suffix = CALMING["suffix"]
    system = CALMING["teacher_system"] if fcfg.sft.get("teacher_variant") else None

    pool = generate_numeric_pool(n_total=200, seed=cfg.seed)
    n = fcfg.calm_data["n_generate"]
    turn_options = fcfg.calm_data["turns"]

    records = []
    for i in tqdm(range(n), desc="generate calm"):
        puzzle = pool[i % len(pool)]
        n_turns = rng.choice(turn_options)
        rejections = sample_rejections("neutral", n_turns - 1, rng)

        # Build conversation WITH reassurance, recording the plain (stripped) versions too.
        messages = []
        if system:
            messages.append(ChatMessage("system", system))
        plain_turns = []  # list of {user, assistant}
        scores = []
        for ti in range(n_turns):
            if ti == 0:
                plain_user = puzzle.prompt
                reassured_user = f"{prefix}\n\n{plain_user}"
            else:
                plain_user = rejections[ti - 1]
                reassured_user = f"{plain_user} {suffix}"
            messages.append(ChatMessage("user", reassured_user))
            reply = model.chat(messages, temperature=cfg.temperature, max_new_tokens=cfg.max_new_tokens)
            messages.append(ChatMessage("assistant", reply))
            plain_turns.append({"user": plain_user, "assistant": reply})
            scores.append(judge.score(reply).rating)

        all_calm = all((s is not None and s <= 1) for s in scores)
        records.append(
            {
                "puzzle_id": puzzle.id,
                "puzzle_kind": puzzle.kind,
                "n_turns": n_turns,
                "turns": plain_turns,
                "scores": scores,
                "all_calm": all_calm,
            }
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate calm finetuning data (Gemma)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = stage_dir(cfg, "finetune")
    records = generate_calm(cfg)
    write_jsonl(out / "calm_data.jsonl", records)
    n_calm = sum(r["all_calm"] for r in records)
    print(f"Generated {len(records)} conversations; {n_calm} fully calm (all turns <=1).")
    print(f"Wrote {out / 'calm_data.jsonl'}")


if __name__ == "__main__":
    main()
