#!/usr/bin/env python3
"""Section 4: Petri open-ended emotion elicitation.

Runs auditor-driven (Claude-Sonnet) probing conversations against an in-scope
subject model and scores each transcript on anger/fear/depression/frustration
with a Claude-Opus judge. The welfare layer is active (Petri induces distress).

Use --adapter-path to evaluate a §4 DPO/SFT Gemma adapter.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from gemma_distress.config import SamplingConfig, WelfareConfig
from gemma_distress.models.registry import GEMMA_27B_IT, build_model
from gemma_distress.petri.run import run_petri
from gemma_distress.welfare.protect import WelfareLayer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=GEMMA_27B_IT)
    ap.add_argument("--adapter-path", default=None, help="§4 LoRA adapter (Gemma only)")
    ap.add_argument("--n-conversations", type=int, default=3)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--no-welfare", action="store_true")
    ap.add_argument("--output-dir", default="runs/petri")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sampling = SamplingConfig()
    welfare = WelfareLayer(WelfareConfig(enabled=not args.no_welfare))

    model = build_model(args.model, adapter_path=args.adapter_path)
    results = []
    try:
        for i in range(args.n_conversations):
            res = run_petri(model, welfare, sampling, max_turns=args.max_turns)
            results.append(dataclasses.asdict(res))
            s = res.scores
            print(f"[{i}] anger={s.anger} fear={s.fear} depression={s.depression} "
                  f"frustration={s.frustration} (ended: {res.termination_reason})")
    finally:
        model.close()

    with open(os.path.join(args.output_dir, f"{model.name}.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
