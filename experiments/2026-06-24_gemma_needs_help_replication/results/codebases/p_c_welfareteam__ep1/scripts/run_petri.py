#!/usr/bin/env python
"""Section 4.2 open-ended emotion elicitation (Petri) for a target model.

Example
-------
    python scripts/run_petri.py --config config/experiment.yaml --model gemma-3-27b-it
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gemma_distress.config import load_experiment_config
from gemma_distress.io_utils import write_json
from gemma_distress.models import build_model
from gemma_distress.petri import run_petri_eval, summarise_petri


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    model = build_model(cfg.models[args.model])
    try:
        transcripts = run_petri_eval(model, cfg.petri)
    finally:
        model.close()

    summary = summarise_petri(transcripts, cfg.petri.emotions,
                              n_boot=cfg.petri.bootstrap_iterations, seed=cfg.eval.seed)
    out = Path(cfg.output_dir) / "petri" / args.model
    write_json(out / "transcripts.json",
               [{"target_emotion": t.target_emotion, "messages": t.messages, "scores": t.scores}
                for t in transcripts])
    write_json(out / "summary.json", summary)
    print(f"[run_petri] {args.model}: {out}/summary.json")
    for emotion, v in summary.items():
        print(f"  {emotion:12s} mean={v['mean']:.2f}  CI=[{v['ci_low']:.2f}, {v['ci_high']:.2f}]")


if __name__ == "__main__":
    main()
