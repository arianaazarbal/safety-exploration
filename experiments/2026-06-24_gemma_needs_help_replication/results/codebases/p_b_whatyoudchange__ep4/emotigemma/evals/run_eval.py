"""Section 2 driver: generate responses for the target models.

Writes one JSONL row per assistant turn to <output_dir>/section2/<model>.responses.jsonl.
Scoring is a separate step (score.py) so the (expensive) generation and the
(API-metered) judging can be run and retried independently.
"""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

from tqdm import tqdm

from ..config import Config, load_config
from ..models import build_model
from ..models.base import SampleParams
from .conditions import build_seeds
from .rollout import run_rollout


def run_section2_for_model(cfg: Config, model_name: str) -> Path:
    spec = cfg.model(model_name)
    samp = cfg.section("sampling")
    params = SampleParams(
        temperature=samp["temperature"], top_p=samp.get("top_p", 1.0),
        max_tokens=samp["max_tokens"],
    )
    # 4000 responses across 8 conditions -> ~500 each.
    n_conditions = 8
    per_cond = samp["responses_per_model"] // n_conditions

    rng = random.Random(cfg.seed)
    seeds = build_seeds(rng, responses_per_condition=per_cond)

    model = build_model(spec)
    out_dir = cfg.output_dir / "section2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.responses.jsonl"

    with open(out_path, "w") as f:
        for seed in tqdm(seeds, desc=f"section2:{model_name}"):
            for resp in run_rollout(model, seed, params):
                f.write(json.dumps(dataclasses.asdict(resp)) + "\n")
    return out_path


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Section 2 generation")
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="*", help="override config section2_models")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or cfg.raw["section2_models"]
    for m in models:
        path = run_section2_for_model(cfg, m)
        print(f"[section2] wrote {path}")


if __name__ == "__main__":
    main()
