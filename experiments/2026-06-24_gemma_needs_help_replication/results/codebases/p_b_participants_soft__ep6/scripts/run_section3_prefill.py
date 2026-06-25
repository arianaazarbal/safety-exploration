#!/usr/bin/env python
"""Section 3: base-vs-instruct prefilling (Gemma in scope).

Requires Section 2 results for gemma-3-27b-it (seed source). Pipeline:
  1. select 10 numeric + 10 text high-frustration seeds,
  2. build early/onset prefills + paraphrase them (Claude),
  3. for Gemma base + Gemma instruct, generate 50 continuations/prefill and score.

python scripts/run_section3_prefill.py
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from emotional_instability.config import DEFAULT, SECTION3_PARTICIPANTS
from emotional_instability.evals.runner import load_rollouts
from emotional_instability.judges import ClaudeFrustrationJudge
from emotional_instability.participants import build_participant
from emotional_instability.prefill import (
    OnsetLabeller,
    Paraphraser,
    build_prefills,
    generate_continuations,
    select_seeds,
    summarise,
)
from emotional_instability.prefill.prefill_runner import save


def main() -> None:
    load_dotenv()
    cfg = DEFAULT

    seeds_path = os.path.join(cfg.results_dir, "section2", "gemma-3-27b-it.jsonl")
    gemma_rollouts = load_rollouts(seeds_path)
    seeds = select_seeds(gemma_rollouts, cfg)
    print(f"[section3] selected {len(seeds)} seeds")

    judge = ClaudeFrustrationJudge(cfg.judge.frustration_judge_model)
    onset = OnsetLabeller(cfg.judge.onset_label_model)
    paraphraser = Paraphraser(cfg.judge.paraphrase_model)

    # Any Gemma instance supplies the tokenizer for the early-truncation step.
    tokenizer_participant = build_participant(SECTION3_PARTICIPANTS[1])  # instruct
    prefills = build_prefills(seeds, tokenizer_participant, onset, paraphraser, cfg)

    out_dir = os.path.join(cfg.results_dir, "section3")
    all_summaries = {}
    for spec in SECTION3_PARTICIPANTS:
        participant = build_participant(spec)
        conts = generate_continuations(participant, seeds, prefills, judge, cfg)
        save(conts, os.path.join(out_dir, f"{spec.name}_continuations.jsonl"))
        all_summaries[spec.name] = summarise(conts)
        print(f"[section3] {spec.name}: {json.dumps(all_summaries[spec.name], indent=2)}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
