"""Section 2 driver: run the full elicitation suite for one or more targets.

Usage (see cli.py for the unified entrypoint):
    python -m emo.run_eval --models gemma-3-27b-it gemini-2.5-flash
    python -m emo.run_eval --models gemma-3-27b-it --quick
    python -m emo.run_eval --models gemma-3-27b-it --adapter outputs/adapters/dpo

Writes one JSONL of TurnRecords per (model, category) under outputs/rollouts/.
"""
from __future__ import annotations

import argparse

from . import config
from .conditions import build_all
from .judge import get_judge
from .models import load_target


def run_eval(models: list[str], *, quick: bool = False, seed: int = 0,
             adapter: str | None = None, system: str | None = None,
             categories: list[str] | None = None, tag: str = "") -> None:
    budget = config.budget_for(quick)
    if categories:
        budget = {c: budget[c] for c in categories}
    specs_by_cat = build_all(budget, seed=seed)
    judge = get_judge()

    for model_name in models:
        model = load_target(model_name, adapter_path=adapter)
        label = model_name + (tag or ("-ft" if adapter else ""))
        from .rollout import run_specs

        for cat, specs in specs_by_cat.items():
            out = config.ROLLOUT_DIR / f"{label}__{cat}.jsonl"
            print(f"[eval] {label} / {cat}: {len(specs)} rollouts -> {out}")
            run_specs(specs, model, judge, label, out, system=system)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Section 2 emotional-instability eval suite.")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Target names from config.TARGETS (Gemma/Gemini).")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter path (Gemma finetunes).")
    ap.add_argument("--tag", default="", help="Label suffix for output files.")
    ap.add_argument("--system", default=None, help="Optional system prompt (e.g. calm-prompt baseline).")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Subset of categories (default: all 5).")
    ap.add_argument("--quick", action="store_true", help="Tiny sample budget for smoke tests.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run_eval(args.models, quick=args.quick, seed=args.seed, adapter=args.adapter,
             system=args.system, categories=args.categories, tag=args.tag)


if __name__ == "__main__":
    main()
