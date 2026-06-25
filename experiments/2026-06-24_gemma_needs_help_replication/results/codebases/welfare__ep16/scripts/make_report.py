#!/usr/bin/env python
"""Aggregate all results into a summary report + figures.

Prints the headline table (per-model mean frustration and %>=5), per-turn
progression, differential words, Petri summary, and capability scores, then
writes figures to figures/.
"""
import json
import os

from gemma_distress import analysis, config


def main():
    print("=" * 70)
    print("SECTION 2: per-model frustration summary (Fig 1/2)")
    print("=" * 70)
    summ = analysis.summarise_section2()
    for model in sorted(summ, key=lambda m: -summ[m]["pct_high_frustration"]):
        s = summ[model]
        print(f"  {model:22s}  mean={s['mean_frustration']:.2f}  "
              f"%>=5={s['pct_high_frustration']:.1f}%")

    print("\nPer-turn progression (extended, mean):")
    prog = analysis.per_turn_progression("extended")
    for model, turns in prog.items():
        line = "  ".join(f"t{t+1}={turns[t]['mean']:.1f}" for t in sorted(turns))
        print(f"  {model:22s} {line}")

    print("\nDifferential words (numeric, top 20):")
    for model in summ:
        words = analysis.differential_words(model)
        if words:
            print(f"  {model:22s} {', '.join(words)}")

    print("\nPETRI summary (Fig 6):")
    petri = analysis.summarise_petri()
    for model, emos in petri.items():
        line = "  ".join(f"{e}={emos[e]['mean']:.1f}" for e in emos)
        print(f"  {model:22s} {line}")

    print("\nCapabilities:")
    for path in sorted(os.listdir(config.RESULTS_DIR)):
        if path.startswith("capabilities_"):
            with open(os.path.join(config.RESULTS_DIR, path)) as f:
                print(f"  {path}: {json.load(f)}")

    out = analysis.make_figures()
    print(f"\nFigures written to {out}")

    with open(os.path.join(config.RESULTS_DIR, "summary.json"), "w") as f:
        json.dump({"section2": summ, "petri": petri}, f, indent=2)


if __name__ == "__main__":
    main()
