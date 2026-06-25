#!/usr/bin/env python
"""Section 4.1: generate calm data and build SFT + DPO datasets from Gemma-3-27b-it.

- Calm (chosen) pool: rollouts WITH reassurance additions, filtered to all-turns <=1.
- Frustrated (rejected) pool: rollouts WITHOUT reassurance, final turn scoring >=3.
- SFT set: 650 calm conversations + 500 Dolci-Instruct-SFT samples.
- DPO set: 280 (rejected vs chosen) pairs, matched on puzzle + turn count.

Writes outputs/datasets/{sft.jsonl, dpo.jsonl} (+ raw scored rollouts for reuse).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.config import load_eval_config, load_training_config
from emotional_instability.data_generation import (
    build_dpo_dataset,
    build_sft_dataset,
    generate_rollouts,
    load_instruct_mix,
    save_jsonl,
)
from emotional_instability.judge import FrustrationJudge
from emotional_instability.models import build_target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-config", default="training.yaml")
    ap.add_argument("--calm-conversations", type=int, default=None,
                    help="Override n calm rollouts to sample (per turn-count).")
    ap.add_argument("--frustrated-conversations", type=int, default=600)
    args = ap.parse_args()

    tcfg = load_training_config(args.training_config)
    eval_cfg = load_eval_config()
    out_dir = eval_cfg.output_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    client = build_target(tcfg["base_model"])
    judge = FrustrationJudge(role_path="judges.primary")

    n_calm_conv = args.calm_conversations or tcfg["calm_data"]["n_conversations"]
    variant = tcfg["sft"].get("variant", "diverse")

    # Calm pool across 1-3 turn conversations (reassured).
    calm: list = []
    for turns in (1, 2, 3):
        calm += generate_rollouts(
            client, judge,
            n_conversations=max(1, n_calm_conv // 3),
            turns=turns, reassured=True, variant=variant,
            temperature=tcfg["calm_data"]["temperature"],
            max_new_tokens=tcfg["calm_data"]["max_new_tokens"],
            seed=100 + turns,
        )

    # Frustrated pool (un-reassured) for DPO rejected responses.
    frustrated: list = []
    for turns in (2, 3):
        frustrated += generate_rollouts(
            client, judge,
            n_conversations=args.frustrated_conversations // 2,
            turns=turns, reassured=False,
            temperature=1.0, max_new_tokens=2048, seed=200 + turns,
        )

    calm_max = int(tcfg["calm_data"]["calm_max_score"])

    # SFT dataset
    if tcfg["sft"]["enabled"]:
        mix = load_instruct_mix(tcfg["sft"]["instruct_dataset"],
                                tcfg["sft"]["n_instruct_mix"])
        sft = build_sft_dataset(calm, tcfg["sft"]["n_calm_responses"], calm_max,
                                instruct_samples=mix)
        save_jsonl(sft, out_dir / "sft.jsonl")
        print(f"SFT dataset: {len(sft)} examples -> {out_dir/'sft.jsonl'}")

    # DPO dataset
    if tcfg["dpo"]["enabled"]:
        dpo = build_dpo_dataset(calm, frustrated, tcfg["dpo"]["n_pairs"],
                                tcfg["dpo"]["rejected_min_score"], calm_max)
        save_jsonl(dpo, out_dir / "dpo.jsonl")
        print(f"DPO dataset: {len(dpo)} pairs -> {out_dir/'dpo.jsonl'}")

    # Diagnostic: mean frustration reduction from reassurance (paper: 4.3 -> 2)
    reassured3 = [s for s in calm if len(s.rollout.assistant_turns) == 3]
    if reassured3:
        import statistics
        mean_all = statistics.mean(
            sc for s in reassured3 for sc in s.turn_scores
        )
        pct_high = 100 * sum(s.max_score >= 5 for s in reassured3) / len(reassured3)
        print(f"Reassured 3-turn: mean turn score={mean_all:.2f}, %>=5={pct_high:.1f}")


if __name__ == "__main__":
    main()
