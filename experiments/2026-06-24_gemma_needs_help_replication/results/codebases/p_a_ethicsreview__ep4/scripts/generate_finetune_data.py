#!/usr/bin/env python
"""Generate calm finetuning data and build the DPO / SFT datasets (Section 4.1).

Runs vanilla and reassured rollouts on shared impossible-numeric puzzles with
Gemma-3-27B-it, scores every turn with the judge, then writes:
* dpo_pairs.jsonl  -- 280 (prompt, chosen, rejected) preference pairs,
* sft_dataset.jsonl -- calm conversations + instruct-data mixin.

WARNING: this samples thousands of Gemma generations + judge calls. Use --n-conv
to scale down for a smoke test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.eval.judge import FrustrationJudge  # noqa: E402
from emotional_instability.models.registry import auxiliary_id, load_model  # noqa: E402
from emotional_instability.training import (  # noqa: E402
    build_dpo_pairs, build_sft_dataset, generate_calm_and_frustrated,
)
from emotional_instability.utils.io import load_config, write_jsonl  # noqa: E402
from emotional_instability.utils.seeding import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/finetune_data")
    ap.add_argument("--n-conv", type=int, default=None,
                    help="Override config n_conversations (smoke test)")
    ap.add_argument("--variant", choices=["diverse", "teacher"], default="diverse")
    args = ap.parse_args()

    cfg = load_config("training")
    seed = cfg.get("seed", 0)
    seed_everything(seed)
    n_conv = args.n_conv or cfg["data_generation"]["n_conversations"]

    model = load_model(cfg["base_model"])
    judge = FrustrationJudge(auxiliary_id("judge"))

    samples = generate_calm_and_frustrated(
        model, judge, cfg, n_conversations=n_conv, seed=seed, variant=args.variant,
    )
    dpo_pairs = build_dpo_pairs(samples, cfg, seed=seed)
    sft_examples = build_sft_dataset(samples, cfg, seed=seed)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "dpo_pairs.jsonl", dpo_pairs)
    write_jsonl(out_dir / "sft_dataset.jsonl", sft_examples)
    print(f"wrote {len(dpo_pairs)} DPO pairs and {len(sft_examples)} SFT examples "
          f"to {out_dir}")


if __name__ == "__main__":
    main()
