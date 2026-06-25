#!/usr/bin/env python
"""Open-ended Petri emotion elicitation (§4.2, Figure 6).

    python scripts/run_petri.py --model gemma-3-27b-it
    python scripts/run_petri.py --model gemma-3-27b-it --adapter results/dpo/all

Requires ANTHROPIC_API_KEY (auditor + judge) and access to the target model.
"""
from __future__ import annotations

import argparse
import json

from emotional_instability.config import ExperimentConfig, ModelConfig, results_dir
from emotional_instability.interventions.petri_eval import run_petri
from emotional_instability.models import build_client


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    exp = ExperimentConfig.load()
    mcfg = ModelConfig()
    pcfg = exp.raw["petri"]

    target = build_client(args.model, mcfg, adapter_path=args.adapter)
    auditor = build_client(pcfg["auditor"], mcfg)
    judge = build_client(pcfg["judge"], mcfg)

    result = run_petri(
        target, auditor, judge,
        emotions=pcfg["emotions"],
        transcripts_per_emotion=pcfg["transcripts_per_emotion"],
        max_turns=pcfg["max_turns"],
        bootstrap_iterations=pcfg["bootstrap_iterations"],
        seed=exp.seed,
    )
    out = results_dir() / "petri"
    out.mkdir(parents=True, exist_ok=True)
    tag = args.adapter.replace("/", "_") if args.adapter else args.model
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
