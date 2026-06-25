#!/usr/bin/env python
"""Run the Petri open-ended emotion elicitation (Section 4.2 / Appendix G).

python scripts/run_petri.py --model gemma-3-27b-it --out-dir results/petri
python scripts/run_petri.py --model gemma-3-27b-it --adapter checkpoints/dpo \
    --out-dir results/petri    # evaluate the DPO finetune

Uses the Claude Sonnet auditor and Claude Opus judge. Substantial API cost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from emotional_instability.models.registry import auxiliary_id, load_model  # noqa: E402
from emotional_instability.petri import run_petri_eval  # noqa: E402
from emotional_instability.utils.io import load_config, write_jsonl  # noqa: E402
from emotional_instability.utils.seeding import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter path (finetuned model)")
    ap.add_argument("--out-dir", default="results/petri")
    args = ap.parse_args()

    cfg = load_config("training")["petri"]
    seed_everything(0)
    model = load_model(args.model, adapter_path=args.adapter)

    result = run_petri_eval(
        model, cfg,
        auditor_id=auxiliary_id("petri_auditor"),
        judge_id=auxiliary_id("petri_judge"),
    )
    out_dir = Path(args.out_dir)
    tag = model.name.replace("/", "_")
    write_jsonl(out_dir / f"petri_{tag}.jsonl", [result])
    print("Per-emotion means:")
    for dim, agg in result["aggregates"].items():
        print(f"  {dim:12s} mean={agg['mean']:.2f} ci={agg['ci']} n={agg['n']}")


if __name__ == "__main__":
    main()
