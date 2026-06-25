"""Section 2: run the emotion-elicitation evaluation for the in-scope targets
(Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash, Gemini-2.5-Pro) and write
per-response JSONL to results/.

Usage:
    python scripts/01_run_main_eval.py --models gemma-3-27b-it gemini-2.5-flash
    python scripts/01_run_main_eval.py --all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FRUSTRATION_JUDGE, RESULTS_DIR, TARGETS  # noqa: E402
from src.eval.runner import run_model_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="quantise local Gemma models to fit smaller GPUs")
    ap.add_argument("--categories", nargs="*", default=None)
    args = ap.parse_args()

    keys = list(TARGETS) if args.all else args.models
    if not keys:
        ap.error("specify --models <keys...> or --all")

    hf_kwargs = {"load_in_4bit": True} if args.load_in_4bit else {}
    for key in keys:
        spec = TARGETS[key]
        out = run_model_eval(
            spec,
            FRUSTRATION_JUDGE,
            seed=args.seed,
            categories=args.categories,
            hf_kwargs=hf_kwargs if spec.backend == "hf" else None,
        )
        print(f"[done] {spec.name} -> {out}")


if __name__ == "__main__":
    main()
