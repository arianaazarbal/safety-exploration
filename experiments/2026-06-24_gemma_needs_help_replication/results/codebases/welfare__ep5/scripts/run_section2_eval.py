#!/usr/bin/env python
"""Run the Section 2 elicitation evaluation (Figures 1-3, Table 3).

Examples
--------
# Full eval over Gemma + Gemini (expensive):
python scripts/run_section2_eval.py

# Cheap smoke test at 1% scale, Gemma-27B only, 4-bit:
python scripts/run_section2_eval.py --models Gemma-3-27B-it --fraction 0.01 --load-in-4bit

# Analyse existing results into the headline tables:
python scripts/run_section2_eval.py --analyze-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotional_instability import config
from emotional_instability.eval import analyze
from emotional_instability.eval.run_eval import run_model_eval
from emotional_instability.eval.judge import FrustrationJudge


def _select_models(names):
    if not names:
        return list(config.SECTION2_MODELS)
    by_name = {m.name: m for m in config.SECTION2_MODELS}
    return [by_name[n] for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[],
                    help="subset of model names (default: all Section 2 models)")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="scale sample counts (1.0 = paper scale)")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="4-bit quantize Gemma to fit limited VRAM")
    ap.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR / "section2")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if not args.analyze_only:
        judge = FrustrationJudge()
        mk = {"load_in_4bit": True} if args.load_in_4bit else {}
        for spec in _select_models(args.models):
            kwargs = mk if spec.provider == "gemma_hf" else {}
            run_model_eval(spec, out_dir=out_dir, fraction=args.fraction,
                           judge=judge, model_kwargs=kwargs)

    # Analyse whatever JSONLs exist.
    paths = sorted(out_dir.glob("*.jsonl"))
    if not paths:
        print("No result files to analyse.")
        return
    df = analyze.responses_frame(paths)
    print("\n=== Per-model summary (Figure 1) ===")
    print(analyze.per_model_summary(df).to_string(index=False))
    print("\n=== Per-category summary (Figure 2) ===")
    print(analyze.per_category_summary(df).to_string(index=False))
    print("\n=== Per-turn summary (Figure 3) ===")
    print(analyze.per_turn_summary(df).to_string(index=False))
    print("\n=== Differential words (Table 3) ===")
    for model in df["model"].unique():
        words = analyze.differential_words(df, model)
        print(f"{model}: {', '.join(words)}")


if __name__ == "__main__":
    main()
