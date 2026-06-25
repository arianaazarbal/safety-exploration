"""Section 4: Petri open-ended emotion elicitation for one or more models.

Example:
    python -m emotional_instability.scripts.run_petri \
        --model gemma-3-27b-it gemma-3-27b-it-dpo gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import json

from ..config import load_config
from ..petri.petri_runner import run_petri_for_model


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for model in args.model:
        path = run_petri_for_model(model, cfg=cfg, seed=args.seed)
        summary = (path.parent / "summary.json").read_text()
        print(f"{model}: {summary}")


if __name__ == "__main__":
    main()
