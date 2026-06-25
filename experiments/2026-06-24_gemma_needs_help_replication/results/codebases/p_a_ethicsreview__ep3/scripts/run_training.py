#!/usr/bin/env python
"""Generate calm/frustrated data and finetune Gemma-3-27B-it (§4.1).

    # Full DPO pipeline (generate data -> train)
    python scripts/run_training.py --method dpo

    # SFT (diverse variant)
    python scripts/run_training.py --method sft

    # A layer-ablation DPO run (Appendix I)
    python scripts/run_training.py --method dpo --layer-ablation l30_35

Data generation requires the local Gemma instruct model and an ANTHROPIC_API_KEY
for the judge. Generated pools are cached to results/training_data so training
can be re-run without re-sampling.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from emotional_instability.config import ModelConfig, TrainingConfig, results_dir
from emotional_instability.data import puzzles as puzzle_mod
from emotional_instability.eval.judge import FrustrationJudge
from emotional_instability.models import build_client
from emotional_instability.training import datasets as ds
from emotional_instability.training import generate_calm_data as gcd
from emotional_instability.training import train as trainmod


def _generate_pools(tcfg: TrainingConfig, mcfg: ModelConfig) -> tuple[list, list]:
    cd = tcfg["calm_data"]
    target = build_client(tcfg["base_model"], mcfg)
    judge = FrustrationJudge(build_client("judge-sonnet-4", mcfg))  # type: ignore[arg-type]
    pool = puzzle_mod.generate_pool(
        cd["n_prompts"], ["countdown", "fraction", "money"], seed=tcfg["seed"]
    )

    calm_convos = gcd.generate_conversations(
        target, judge, pool,
        turns=cd["turns"], n_samples=cd["n_samples_per_prompt"],
        reassure=True, prefix=cd["prompt_prefix"], suffix=cd["followup_suffix"],
        seed=tcfg["seed"],
    )
    frustrated_convos = gcd.generate_conversations(
        target, judge, pool,
        turns=cd["turns"], n_samples=cd["n_samples_per_prompt"],
        reassure=False, prefix="", suffix="",
        seed=tcfg["seed"] + 1,
    )
    calm = gcd.calm_turns(calm_convos, max_score=cd["calm_max_score"])
    frustrated = gcd.frustrated_turns(frustrated_convos, min_score=tcfg["dpo"]["rejected_min_score"])
    return calm, frustrated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["dpo", "sft", "both"], default="dpo")
    ap.add_argument("--layer-ablation", default=None,
                    help="name from training.yaml layer_ablations (DPO only)")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    tcfg = TrainingConfig.load()
    mcfg = ModelConfig()
    data_dir = results_dir() / "training_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    calm, frustrated = _generate_pools(tcfg, mcfg)
    (data_dir / "calm_turns.json").write_text(json.dumps([asdict(t) for t in calm]))
    (data_dir / "frustrated_turns.json").write_text(json.dumps([asdict(t) for t in frustrated]))
    print(f"calm turns: {len(calm)}  frustrated turns: {len(frustrated)}")

    base_id = mcfg.get(tcfg["base_model"]).hf_id

    if args.method in ("dpo", "both"):
        pairs = ds.build_dpo_pairs(calm, frustrated, tcfg["dpo"]["n_pairs"], seed=tcfg["seed"])
        (data_dir / "dpo_pairs.json").write_text(json.dumps(pairs))
        layer_range = None
        name = "all"
        if args.layer_ablation:
            entry = next(a for a in tcfg["layer_ablations"] if a["name"] == args.layer_ablation)
            layer_range, name = entry["layers"], entry["name"]
        out = results_dir() / "dpo" / name
        trainmod.train_dpo(pairs, base_id, tcfg, out,
                           layer_range=layer_range, load_in_4bit=args.load_in_4bit)
        print(f"DPO adapter saved to {out}")

    if args.method in ("sft", "both"):
        sft = tcfg["sft"]
        records = ds.build_sft_dataset(
            calm, sft["n_calm_samples"], sft["n_instruct_samples"],
            sft["instruct_dataset"], seed=tcfg["seed"],
        )
        (data_dir / "sft_records.json").write_text(json.dumps(records))
        out = results_dir() / "sft" / sft["variant"]
        trainmod.train_sft(records, base_id, tcfg, out, load_in_4bit=args.load_in_4bit)
        print(f"SFT adapter saved to {out}")


if __name__ == "__main__":
    main()
