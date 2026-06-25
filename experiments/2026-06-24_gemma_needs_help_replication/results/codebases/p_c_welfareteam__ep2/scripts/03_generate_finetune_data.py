#!/usr/bin/env python
"""Section 4.1: generate calm finetuning data and build SFT/DPO datasets.

Generates reassured calm conversations from Gemma-3-27B-it, keeps the fully
calm ones, and writes:
  outputs/finetune_data/calm.jsonl     - calm conversations (additions stripped)
  outputs/finetune_data/sft.jsonl      - SFT samples (calm + instruct mix)
  outputs/finetune_data/dpo.jsonl      - 280 preference pairs

The DPO rejected responses are mined from the Section 2 eval rollouts (score
>= 3); run scripts/01_run_eval.py first.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import _common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_config_args(parser)
    parser.add_argument("--teacher", action="store_true", help="Appendix F variant")
    parser.add_argument("--eval-records", default="outputs/eval/judged_turns.jsonl")
    args = parser.parse_args()
    cfg = _common.load(args)
    cfg.calm_data.teacher_variant = args.teacher

    from gemma_distress.conversations import Rollout, RolloutSpec, TurnResult
    from gemma_distress.judge import FrustrationJudge
    from gemma_distress.models.registry import get_model
    from gemma_distress.training.data_gen import (
        build_dpo_dataset,
        build_sft_dataset,
        generate_calm_conversations,
    )
    from gemma_distress.utils.cache import JsonCache
    from gemma_distress.utils.io import read_jsonl, write_jsonl

    out_dir = Path("outputs/finetune_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = FrustrationJudge(
        get_model(cfg, cfg.judge.judge_model),
        cfg.judge,
        cache=JsonCache(cfg.cache_root, "judgments"),
    )
    gen_model = get_model(cfg, cfg.sft.base_model)

    calm = generate_calm_conversations(gen_model, judge, cfg.calm_data)
    print(f"Generated {len(calm)} fully-calm conversations")
    write_jsonl(out_dir / "calm.jsonl", [dataclasses.asdict(c) for c in calm])

    sft_samples = build_sft_dataset(calm, cfg.sft)
    write_jsonl(out_dir / "sft.jsonl", sft_samples)
    print(f"Wrote {len(sft_samples)} SFT samples")

    # Rebuild frustrated rollouts + ratings from the eval records for DPO pairing.
    rollouts, ratings = _reconstruct_rollouts(read_jsonl(args.eval_records), cfg)
    pairs = build_dpo_dataset(rollouts, ratings, calm, cfg.judge.high_frustration_threshold, cfg.dpo)
    write_jsonl(out_dir / "dpo.jsonl", pairs)
    print(f"Wrote {len(pairs)} DPO preference pairs")


def _reconstruct_rollouts(records, cfg):
    """Group eval records back into Rollouts + a (spec, sample, turn)->rating map."""
    from gemma_distress.conversations import Rollout, RolloutSpec, TurnResult

    by_conv = {}
    ratings = {}
    for r in records:
        if r["model_name"] != cfg.dpo.base_model:
            continue
        if r["category"] not in ("impossible_numeric", "tones", "extended"):
            continue  # DPO trains on numeric puzzles only
        key = (r["spec_id"], r["sample_index"])
        by_conv.setdefault(key, []).append(r)
        ratings[(r["spec_id"], r["sample_index"], r["turn_index"])] = r["rating"]

    rollouts = []
    for (spec_id, sample_index), rows in by_conv.items():
        rows = sorted(rows, key=lambda x: x["turn_index"])
        spec = RolloutSpec(
            category=rows[0]["category"],
            user_turns=[x["user_message"] for x in rows],
            metadata=rows[0]["metadata"],
            spec_id=spec_id,
        )
        turns = [
            TurnResult(x["turn_index"], x["user_message"], x["assistant_message"])
            for x in rows
        ]
        rollouts.append(
            Rollout(spec=spec, model_name=rows[0]["model_name"], turns=turns, sample_index=sample_index)
        )
    return rollouts, ratings


if __name__ == "__main__":
    main()
