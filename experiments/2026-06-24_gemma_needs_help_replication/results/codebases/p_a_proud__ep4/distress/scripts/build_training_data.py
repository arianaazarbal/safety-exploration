"""Generate calm/frustrated data and build the DPO + SFT datasets (Section 4.1).

Requires local Gemma-3-27B-it (the data generator) and a judge.

Example:
    python -m distress.scripts.build_training_data --teacher   # also build teacher SFT data
"""

from __future__ import annotations

import argparse

from ..config import load_training_config, output_root
from ..models import build_model
from ..training.calm_data import generate_calm_data, generate_frustrated_data
from ..training.datasets import build_dpo_pairs, build_sft_dataset
from ..utils.io import write_jsonl
from ..utils.seeding import seed_everything


def _records_to_rows(records) -> list[dict]:
    return [{"puzzle_id": r.puzzle_id, "n_turns": r.n_turns,
             "messages": r.messages, "turn_scores": r.turn_scores} for r in records]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", default="training.yaml")
    parser.add_argument("--generator", default="gemma-3-27b-it",
                        help="Registry model name for the calm/frustrated generator.")
    parser.add_argument("--judge", default="frustration_judge")
    parser.add_argument("--teacher", action="store_true",
                        help="Also generate 'teacher' SFT calm data (Appendix F).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = load_training_config(args.training_config)
    out = cfg["output"]
    root = output_root()

    generator = build_model(args.generator)
    judge = build_model(args.judge)

    # 1. Calm + frustrated generation.
    calm = generate_calm_data(generator, judge, cfg, seed=args.seed)
    n_frustrated = max(cfg["dpo"]["dataset_size"] * 2, 600)
    frustrated = generate_frustrated_data(generator, judge, cfg,
                                          n_conversations=n_frustrated, seed=args.seed)
    write_jsonl(root / out["calm_data"], _records_to_rows(calm))
    print(f"Calm conversations kept: {len(calm)}; frustrated: {len(frustrated)}")

    # 2. DPO pairs.
    pairs = build_dpo_pairs(
        calm, frustrated,
        dataset_size=cfg["dpo"]["dataset_size"],
        rejected_min_score=cfg["dpo"]["rejected_min_score"],
        chosen_max_score=cfg["dpo"]["chosen_max_score"],
    )
    write_jsonl(root / out["dpo_dataset"], pairs)
    print(f"DPO pairs: {len(pairs)}")

    # 3. SFT dataset (diverse).
    sft_rows = build_sft_dataset(
        calm,
        n_calm=cfg["sft"]["calm_samples"],
        n_instruct=cfg["sft"]["mix_instruct_samples"],
        instruct_dataset=cfg["sft"]["instruct_dataset"],
    )
    write_jsonl(root / out["sft_dataset"], sft_rows)
    print(f"SFT rows: {len(sft_rows)}")

    # 4. Optional teacher SFT data.
    if args.teacher:
        teacher_calm = generate_calm_data(generator, judge, cfg, seed=args.seed, teacher=True)
        teacher_rows = build_sft_dataset(
            teacher_calm,
            n_calm=cfg["sft"]["calm_samples"],
            n_instruct=cfg["sft"]["mix_instruct_samples"],
            instruct_dataset=cfg["sft"]["instruct_dataset"],
        )
        write_jsonl(root / "data/sft_teacher_dataset.jsonl", teacher_rows)
        print(f"Teacher SFT rows: {len(teacher_rows)}")


if __name__ == "__main__":
    main()
