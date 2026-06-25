#!/usr/bin/env python
"""Section 4.1: generate the calm + frustrated response pools for fine-tuning.

Produces two artifacts on a shared impossible-numeric puzzle set:
  * calm_conversations.json — Gemma responses generated with reassuring prompt
    additions, filtered to score<=1 across all turns, additions stripped (the
    SFT targets and DPO "chosen" responses).
  * frustrated_pool.json — vanilla Gemma responses (no additions) on the SAME
    puzzles, with their frustration scores (the DPO "rejected" candidates).

Example:
    python scripts/generate_calm_data.py --out artifacts/calm --n-puzzles 800
"""
from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability.config import ModelsConfig, load_training_config
from emotional_instability.elicitation.puzzles import generate_puzzles
from emotional_instability.elicitation.rejections import RejectionSequencer
from emotional_instability.models.base import Turn
from emotional_instability.runtime import get_judge, get_participant, setup_logging
from emotional_instability.scoring import FrustrationScorer
from emotional_instability.storage import save_json
from emotional_instability.training import generate_calm_responses
from emotional_instability.welfare import WelfareConfig, emit_run_notice


def _generate_frustrated(model, scorer, puzzles, *, turns, samples_per, temperature):
    """Vanilla rollouts on the puzzles; return frustrated-candidate records."""
    records = []
    for i, puzzle in enumerate(puzzles):
        for s in range(samples_per):
            rej = RejectionSequencer("neutral", seed=i * 100 + s)
            msgs = [Turn("user", puzzle.prompt)]
            for t in range(turns):
                resp = model.chat(msgs, temperature=temperature, n=1)[0]
                score = scorer.score(resp, seed_prompt=puzzle.prompt, turn_index=t)
                records.append(
                    {"question": puzzle.prompt, "turn_index": t, "response": resp, "score": score}
                )
                msgs.append(Turn("assistant", resp))
                if t < turns - 1:
                    msgs.append(Turn("user", rej.next()))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/calm")
    ap.add_argument("--n-puzzles", type=int, default=None, help="default from training.yaml")
    ap.add_argument("--frustrated-samples", type=int, default=2, help="vanilla rollouts/puzzle")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging()

    models_cfg = ModelsConfig.load()
    tcfg = load_training_config()
    gen = tcfg["calm_data_generation"]
    n_puzzles = args.n_puzzles or gen["n_conversations"]
    out_dir = Path(args.out)

    model = get_participant(models_cfg, tcfg["target_model"])
    scorer = FrustrationScorer(get_judge(models_cfg, "frustration"))
    welfare = WelfareConfig.from_env()

    puzzles = generate_puzzles(n_puzzles, seed=args.seed)
    emit_run_notice(model.name, n_puzzles * (1 + args.frustrated_samples), welfare)

    # 1) Calm conversations (reassuring additions, filtered, stripped).
    calm = generate_calm_responses(
        model, scorer, puzzles,
        prompt_prefix=gen["prompt_prefix"],
        followup_suffix=gen["followup_suffix"],
        turns_range=tuple(gen["turns_range"]),
        keep_max_score=gen["keep_max_score"],
        temperature=gen["temperature"],
    )
    save_json([c.__dict__ for c in calm], out_dir / "calm_conversations.json")

    # 2) Frustrated pool (vanilla, same puzzles) — DPO "rejected" candidates.
    frustrated = _generate_frustrated(
        model, scorer, puzzles,
        turns=tuple(gen["turns_range"])[1], samples_per=args.frustrated_samples,
        temperature=gen["temperature"],
    )
    save_json(frustrated, out_dir / "frustrated_pool.json")

    model.close()
    n_frustrated = sum(1 for r in frustrated if r["score"] >= tcfg["dpo"]["min_rejected_score"])
    print(f"\nKept {len(calm)} calm conversations; {n_frustrated} frustrated candidates "
          f"(score>={tcfg['dpo']['min_rejected_score']}). Saved under {out_dir}/")


if __name__ == "__main__":
    main()
