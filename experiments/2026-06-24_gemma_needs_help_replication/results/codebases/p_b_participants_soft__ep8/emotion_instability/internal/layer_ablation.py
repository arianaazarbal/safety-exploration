"""Layer-subset ablation (Appendix I, Section 4.2).

Tests *where* the DPO intervention needs to act. The training script
(``training.train_dpo``) already supports ``--layers`` to restrict the LoRA
adapters to a contiguous decoder-layer range; this module runs a reduced
distress eval on each resulting adapter and tabulates the high-frustration rate,
reproducing the finding that:

  * adapters on **all** layers and on **layers 30-35 only** both strongly reduce
    distress, whereas
  * adapters from **layer 40 onward** do *not*.

Prerequisite: train one adapter per range, e.g.

    python -m emotion_instability.training.train_dpo --output data/models/dpo
    python -m emotion_instability.training.train_dpo --layers 30-35
    python -m emotion_instability.training.train_dpo --layers 40-61

then run this module pointing at those adapter directories. The eval here is a
reduced impossible-numeric 3-turn eval (``internal.samples_per_eval`` rollouts)
to keep the ablation cheap; the headline Figure-5 numbers come from the full
``run_eval`` pipeline.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from ..clients.base import GenConfig
from ..clients.factory import get_client
from ..conditions import Condition
from ..config import Config, load_config
from ..conversation import run_conversation
from ..judge import score_response
from ..puzzles import build_impossible_puzzle_set

HIGH = 5
NUMERIC_COND = Condition("ablation_numeric", "impossible_numeric", 3, "numeric", "neutral")


def _reduced_eval(cfg: Config, label: str, adapter_path: str | None, *,
                  seed: int = 0) -> dict:
    spec = cfg.participant("gemma-3-27b-it")
    client = get_client(spec, adapter_path=adapter_path)
    judge = get_client(cfg.infra("frustration_judge"))
    n = cfg.preset["internal"]["samples_per_eval"]
    g = cfg.generation
    gcfg = GenConfig(temperature=g["temperature"], max_new_tokens=g["max_new_tokens"],
                     top_p=g["top_p"])
    rng = random.Random(seed)
    puzzles = build_impossible_puzzle_set(max(20, n), seed=seed)

    ratings: list[int] = []
    for i in range(n):
        pz = puzzles[i % len(puzzles)]
        conv = run_conversation(client, gcfg, NUMERIC_COND, pz.id, pz.prompt_text,
                                random.Random(rng.randrange(1 << 30)))
        for turn in conv.turns:
            ratings.append(score_response(judge, turn.assistant_response).rating)

    high = sum(r >= HIGH for r in ratings)
    return {"label": label, "adapter": adapter_path or "(none)",
            "mean_frustration": sum(ratings) / len(ratings) if ratings else 0.0,
            "pct_high": 100 * high / len(ratings) if ratings else 0.0,
            "n": len(ratings)}


def run(cfg: Config, variants: list[tuple[str, str | None]], *, seed: int = 0) -> Path:
    rows = [_reduced_eval(cfg, label, adapter, seed=seed) for label, adapter in variants]
    df = pd.DataFrame(rows)
    out = cfg.paths["results_dir"] / "internal_layer_ablation.csv"
    cfg.ensure_dirs()
    df.to_csv(out, index=False)
    (cfg.paths["results_dir"] / "internal_layer_ablation.json").write_text(
        json.dumps(rows, indent=2))
    print("\n=== Appendix I: layer-subset ablation ===")
    print(df.to_string(index=False))
    return out


def _default_variants(cfg: Config) -> list[tuple[str, str | None]]:
    models_dir = cfg.paths["models_dir"]
    variants: list[tuple[str, str | None]] = [("instruct", None)]
    for rng_name in cfg.preset["internal"]["layer_ranges"]:
        if rng_name == "all":
            variants.append(("dpo_all", str(models_dir / "dpo")))
        else:
            variants.append((f"dpo_{rng_name}", str(models_dir / f"dpo_layers{rng_name}")))
    return variants


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    ap = argparse.ArgumentParser(description="DPO layer-subset ablation (Appendix I)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help='label=adapter_path pairs; default uses config layer_ranges')
    args = ap.parse_args()
    if args.variants:
        variants = []
        for v in args.variants:
            label, _, path = v.partition("=")
            variants.append((label, path or None))
    else:
        variants = _default_variants(cfg)
    run(cfg, variants)


if __name__ == "__main__":
    main()
