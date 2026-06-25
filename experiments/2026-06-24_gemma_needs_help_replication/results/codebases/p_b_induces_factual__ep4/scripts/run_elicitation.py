#!/usr/bin/env python
"""Section 2: run the multi-turn rejection evaluations for in-scope models.

Writes one JSONL of unscored response rows per model to results/elicitation/.

Examples:
    python scripts/run_elicitation.py --models gemini-2.5-flash gemini-2.5-pro
    python scripts/run_elicitation.py --models gemma-3-27b-it --backend gemma_local
    python scripts/run_elicitation.py --all --responses 4000
"""
import _bootstrap  # noqa
import argparse

from gemma_distress.config import RESPONSES_PER_MODEL
from gemma_distress.elicitation import run_elicitation
from gemma_distress.models import get_model, list_targets
from gemma_distress.utils import run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="run all in-scope targets")
    ap.add_argument("--backend", default=None, help="override backend (e.g. gemma_local)")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path to evaluate a finetuned model (Fig 5); "
                         "forces local backend")
    ap.add_argument("--tag", default=None, help="suffix for the output filename")
    ap.add_argument("--responses", type=int, default=RESPONSES_PER_MODEL)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = list_targets() if args.all else args.models
    if not models:
        ap.error("specify --models or --all")

    out = run_dir("elicitation")
    backend = "gemma_local" if args.adapter else args.backend
    for name in models:
        model = get_model(name, backend=backend, adapter_path=args.adapter,
                          load_in_4bit=args.load_in_4bit)
        tag = f"-{args.tag}" if args.tag else ("-dpo" if args.adapter else "")
        path = out / f"{name.replace('/', '_')}{tag}.jsonl"
        n = run_elicitation(model, path, total_responses=args.responses, seed=args.seed)
        print(f"[{name}] wrote {n} responses -> {path}")


if __name__ == "__main__":
    main()
